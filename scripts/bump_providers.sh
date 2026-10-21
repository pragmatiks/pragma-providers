#!/usr/bin/env bash
#
# Bump every provider that has releasable commits on its own paths, relock the workspace, and bundle
# the result for the release push.
#
# Usage:
#   bump_providers.sh <bundle_path> <output_path>
#
# Arguments:
#   bundle_path  File the git bundle is written to when a provider was bumped. It holds the full
#                history of the local main branch and every tag this run created, so a fresh repository
#                can fetch it without other objects.
#   output_path  File the results are appended to as key=value lines, such as $GITHUB_OUTPUT.
#
# Environment:
#   PUBLISH_ONLY     "true" bumps nothing and reports the current version of every provider.
#   ALLOW_NO_COMMIT  "true" bumps every provider as a PATCH even without an eligible commit.
#
# Requires: git with user.name and user.email configured, run on the checked-out main branch;
# git-cliff 2.14.2 and commitizen 4.19.2 on PATH; jq; gh authenticated (GH_TOKEN) with read access to
# the releases of the repository GITHUB_REPOSITORY names as owner/name; uv at the version the
# workspace pyproject.toml pins.
#
# Providers are handled in dependency order (gcp before kubernetes, kubernetes before qdrant and agno),
# so each bump commit sits on top of the previous one. git-cliff, configured by cliff.toml at the
# repository root, reads only the commits that touch the provider's own paths (packages/<provider>,
# plus taskfiles/pragma.Taskfile.yaml for pragma) since the provider's release tag. A provider is
# bumped when at least one of those commits is a feat, fix, refactor or perf commit or a breaking
# change; git-cliff computes the next version from them and prepends their changelog entry to
# packages/<provider>/CHANGELOG.md. A provider that depends on one bumped in this run is bumped as
# well: by its own commits when it has releasable ones, otherwise as a PATCH whose changelog entry has
# only its heading. commitizen then writes the version into the provider's version files, commits and
# creates the release tag. Eligibility is read from main, not from the triggering push, so a run that
# was cancelled or stopped leaves nothing that the next run does not see. A provider with nothing to
# bump whose current release tag has no GitHub Release, neither draft nor published, is reported for
# publishing at its tag, so any later run completes a release that a run failing after its push left
# incomplete; the GitHub Release is created last.
#
# Results, for each provider that was bumped, completed or, with PUBLISH_ONLY, reported:
#   <provider>_version  the version to publish
#   <provider>_commit   the commit its release tag points to
# and once:
#   bumped  "true" when at least one provider was bumped and the bundle was written, else "false";
#           with "false", every reported provider is published from its existing tag without a push
#
# After the bumps, uv lock refreshes the workspace lock; a changed lock is committed as
# "build: relock the provider workspace after version bumps [skip ci]", and the workspace must then
# install with uv sync --locked. Exits 1 when HEAD is not the main branch, when the release tag of a
# provider's current version is missing or is not the latest tag git-cliff finds for its paths from
# main, when git-cliff computes a version that is not greater than the current one, when the GitHub
# GraphQL lookup of a release fails for any reason, or when git-cliff, commitizen, uv or git fails.

set -euo pipefail

FAILURE_EXIT_CODE=1
PROVIDERS=(gcp supabase vercel github pragma kubernetes qdrant agno)
RELOCK_MESSAGE="build: relock the provider workspace after version bumps [skip ci]"

if [ -z "${1:-}" ] || [ -z "${2:-}" ]; then
  echo "bump_providers: usage: bump_providers.sh <bundle_path> <output_path>" >&2
  exit "${FAILURE_EXIT_CODE}"
fi

if [ "$(git symbolic-ref --quiet HEAD || true)" != "refs/heads/main" ]; then
  echo "bump_providers: the checked-out branch must be main" >&2
  exit "${FAILURE_EXIT_CODE}"
fi

BUNDLE_PATH="$1"
OUTPUT_PATH="$2"
GIT_CLIFF_CONFIG="$(git rev-parse --show-toplevel)/cliff.toml"
export GIT_CLIFF_CONFIG

BUMPED_PROVIDERS=" "
CREATED_TAGS=()

read_project_version() {
  (cd "packages/$1" && cz version --project)
}

compute_release_tag() {
  echo "$1-v$2"
}

compute_next_patch_version() {
  local major
  local minor
  local patch

  IFS=. read -r major minor patch <<< "$1"
  echo "${major}.${minor}.$((patch + 1))"
}

compute_dependencies() {
  case "$1" in
    kubernetes) echo "gcp" ;;
    qdrant) echo "kubernetes" ;;
    agno) echo "gcp kubernetes" ;;
  esac
}

compute_include_path_arguments() {
  case "$1" in
    pragma) echo "--include-path packages/pragma/** --include-path taskfiles/pragma.Taskfile.yaml" ;;
    *) echo "--include-path packages/$1/**" ;;
  esac
}

has_bumped_dependency() {
  local dependency

  for dependency in $(compute_dependencies "$1"); do
    case "${BUMPED_PROVIDERS}" in
      *" ${dependency} "*) return 0 ;;
    esac
  done

  return 1
}

run_provider_cliff() {
  local provider="$1"
  shift
  local include_path_arguments

  read -r -a include_path_arguments <<< "$(compute_include_path_arguments "${provider}")"
  git cliff --tag-pattern "^${provider}-v[0-9]" "${include_path_arguments[@]}" "$@"
}

load_unreleased_context() {
  run_provider_cliff "$1" --unreleased --context
}

record_release() {
  local provider="$1"
  local version="$2"
  local commit

  commit="$(git rev-parse "refs/tags/$(compute_release_tag "${provider}" "${version}")^{commit}")"
  echo "${provider}_version=${version}" >> "${OUTPUT_PATH}"
  echo "${provider}_commit=${commit}" >> "${OUTPUT_PATH}"
}

report_current_version() {
  local provider="$1"
  local version
  local tag

  version="$(read_project_version "${provider}")"
  tag="$(compute_release_tag "${provider}" "${version}")"
  if ! git rev-parse --quiet --verify "refs/tags/${tag}" > /dev/null; then
    echo "bump_providers: ${provider}: release tag ${tag} does not exist" >&2
    exit "${FAILURE_EXIT_CODE}"
  fi

  echo "bump_providers: ${provider}: publishing the current version ${version}"
  record_release "${provider}" "${version}"
}

fetch_release_draft_state() {
  gh api graphql \
    -f query="query(\$owner: String!, \$name: String!, \$tag: String!) { repository(owner: \$owner, name: \$name) { release(tagName: \$tag) { isDraft } } }" \
    -f owner="${GITHUB_REPOSITORY%%/*}" \
    -f name="${GITHUB_REPOSITORY#*/}" \
    -f tag="$1" \
    --jq .data.repository.release.isDraft
}

resume_unreleased_provider() {
  local provider="$1"
  local version="$2"
  local tag
  local release_draft_state

  tag="$(compute_release_tag "${provider}" "${version}")"
  release_draft_state="$(fetch_release_draft_state "${tag}")"
  if [ -n "${release_draft_state}" ]; then
    echo "bump_providers: ${provider}: no releasable commit on its paths"
    return 0
  fi

  echo "bump_providers: ${provider}: ${tag} has no GitHub Release yet; completing it"
  record_release "${provider}" "${version}"
}

compute_next_version() {
  local provider="$1"
  local current_version="$2"
  local next_tag
  local next_version

  if [ "${ALLOW_NO_COMMIT:-}" = "true" ]; then
    compute_next_patch_version "${current_version}"
    return 0
  fi

  next_tag="$(run_provider_cliff "${provider}" --bumped-version)"
  next_version="${next_tag#"${provider}-v"}"
  if [ "$(printf '%s\n%s\n' "${current_version}" "${next_version}" | sort -V | tail -n 1)" = "${current_version}" ]; then
    echo "bump_providers: ${provider}: git-cliff computed ${next_tag}, not greater than ${current_version}" >&2
    exit "${FAILURE_EXIT_CODE}"
  fi

  echo "${next_version}"
}

bump_provider() {
  local provider="$1"
  local current_version
  local current_tag
  local context
  local next_version

  current_version="$(read_project_version "${provider}")"
  current_tag="$(compute_release_tag "${provider}" "${current_version}")"
  context="$(load_unreleased_context "${provider}")"
  if [ "$(jq -r '.[0].previous.version // ""' <<< "${context}")" != "${current_tag}" ]; then
    echo "bump_providers: ${provider}: ${current_tag} is not the latest release tag git-cliff finds for its paths" >&2
    exit "${FAILURE_EXIT_CODE}"
  fi

  if [ "$(jq '[.[].commits[]] | length' <<< "${context}")" -gt 0 ] || [ "${ALLOW_NO_COMMIT:-}" = "true" ]; then
    next_version="$(compute_next_version "${provider}" "${current_version}")"
  elif has_bumped_dependency "${provider}"; then
    echo "bump_providers: ${provider}: a dependency was bumped; bumping as a PATCH"
    next_version="$(compute_next_patch_version "${current_version}")"
  else
    resume_unreleased_provider "${provider}" "${current_version}"
    return 0
  fi

  run_provider_cliff "${provider}" --unreleased --tag "$(compute_release_tag "${provider}" "${next_version}")" \
    --prepend "packages/${provider}/CHANGELOG.md" > /dev/null
  git add "packages/${provider}/CHANGELOG.md"
  (cd "packages/${provider}" && cz bump "${next_version}" --yes --allow-no-commit)

  echo "bump_providers: ${provider}: bumped to ${next_version}"
  BUMPED_PROVIDERS="${BUMPED_PROVIDERS}${provider} "
  CREATED_TAGS+=("refs/tags/$(compute_release_tag "${provider}" "${next_version}")")
  record_release "${provider}" "${next_version}"
}

relock_workspace() {
  uv lock
  if ! git diff --quiet -- uv.lock; then
    git commit --quiet --message "${RELOCK_MESSAGE}" -- uv.lock
  fi
  uv sync --all-packages --all-groups --locked
}

for provider in "${PROVIDERS[@]}"; do
  if [ "${PUBLISH_ONLY:-}" = "true" ]; then
    report_current_version "${provider}"
  else
    bump_provider "${provider}"
  fi
done

if [ "${#CREATED_TAGS[@]}" -eq 0 ]; then
  echo "bumped=false" >> "${OUTPUT_PATH}"
  exit 0
fi

relock_workspace
git bundle create "${BUNDLE_PATH}" refs/heads/main "${CREATED_TAGS[@]}"
echo "bumped=true" >> "${OUTPUT_PATH}"
