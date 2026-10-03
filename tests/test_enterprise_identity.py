import hashlib

import pytest
from fastapi import HTTPException

from app.config import Settings
from app.security import authenticate_subject


def test_api_key_fallback_authentication() -> None:
    secret = "service-secret"
    settings = Settings(
        auth_enabled=True,
        api_key_sha256=hashlib.sha256(secret.encode()).hexdigest(),
    )
    assert authenticate_subject(secret, settings) == "api-key"


def test_unconfigured_authentication_rejected() -> None:
    settings = Settings(auth_enabled=True)
    with pytest.raises(HTTPException) as exc:
        authenticate_subject("unknown", settings)
    assert exc.value.status_code == 503
