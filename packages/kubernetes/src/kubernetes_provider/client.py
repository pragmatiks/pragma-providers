"""Kubernetes client building from GKE cluster outputs."""

from __future__ import annotations

import base64
import json
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

from google.auth.transport.requests import Request
from google.oauth2 import credentials as user_credentials
from google.oauth2 import service_account
from lightkube import AsyncClient, KubeConfig


if TYPE_CHECKING:
    from gcp_provider import GKEOutputs


_SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]


def _get_access_token(credentials: dict[str, Any] | str) -> str:
    """Get access token from GCP credentials.

    Supports both service account and authorized_user credential types.

    Args:
        credentials: GCP credentials dict or JSON string.

    Returns:
        Bearer access token for Kubernetes API authentication.
    """
    if isinstance(credentials, str):
        credentials = json.loads(credentials)

    cred_type = credentials.get("type", "service_account")

    if cred_type == "authorized_user":
        creds = user_credentials.Credentials.from_authorized_user_info(
            credentials,
            scopes=_SCOPES,
        )
    else:
        creds = service_account.Credentials.from_service_account_info(
            credentials,
            scopes=_SCOPES,
        )

    creds.refresh(Request())

    return creds.token


def create_client_from_gke(
    outputs: GKEOutputs,
    credentials: dict[str, Any] | str,
) -> AsyncClient:
    """Create a lightkube AsyncClient from GKE cluster outputs.

    Builds a kubeconfig from the GKE cluster endpoint and CA certificate,
    authenticates using a bearer token obtained from the GCP credentials.

    Args:
        outputs: GKE cluster outputs containing endpoint and CA certificate.
        credentials: GCP service account credentials for token acquisition.

    Returns:
        Configured lightkube AsyncClient.
    """
    token = _get_access_token(credentials)

    kubeconfig = {
        "apiVersion": "v1",
        "kind": "Config",
        "clusters": [
            {
                "name": "gke-cluster",
                "cluster": {
                    "server": f"https://{outputs.endpoint}",
                    "certificate-authority-data": outputs.cluster_ca_certificate,
                },
            }
        ],
        "contexts": [
            {
                "name": "gke-context",
                "context": {
                    "cluster": "gke-cluster",
                    "user": "gke-user",
                },
            }
        ],
        "current-context": "gke-context",
        "users": [
            {
                "name": "gke-user",
                "user": {
                    "token": token,
                },
            }
        ],
    }

    config = KubeConfig.from_dict(kubeconfig)

    return AsyncClient(config=config)


def write_ca_cert_file(ca_certificate_base64: str) -> Path:
    """Write CA certificate to a temporary file.

    Args:
        ca_certificate_base64: Base64-encoded CA certificate.

    Returns:
        Path to temporary file containing the decoded CA certificate.
    """
    ca_data = base64.b64decode(ca_certificate_base64)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".crt")
    tmp.write(ca_data)
    tmp.close()

    return Path(tmp.name)
