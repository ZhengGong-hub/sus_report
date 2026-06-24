import requests
import os
import json

from carbontax.utils.logger import Logger

logger = Logger.get("acquire.tokens")

def quick_refresh_and_save_token(token_path="tokens/token_current.json"):
    """Refreshes the access token using stored refresh token and saves new tokens to file."""
    TOKEN_REFRESH_URL = (
        "https://api-ciq.marketintelligence.spglobal.com"
        "/gdsapi/rest/authenticate/api/v1/tokenRefresh"
    )
    # Load current refresh token
    with open(token_path, "r") as f:
        current_tokens = json.load(f)
    refresh_token = current_tokens["refresh_token"]

    # Send refresh request
    payload = {"refreshToken": refresh_token}
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    resp = requests.post(TOKEN_REFRESH_URL, data=payload, headers=headers)
    if resp.status_code != 200:
        raise RuntimeError(f"Token refresh failed: {resp.status_code} - {resp.text}")

    td = resp.json()
    tokens = {
        "access_token": td["access_token"],
        "refresh_token": td.get("refresh_token", refresh_token),
        "expires_in_seconds": int(td["expires_in_seconds"]),
        "token_type": td["token_type"],
        "scope": td.get("scope")
    }

    # Ensure tokens directory exists and save
    os.makedirs(os.path.dirname(token_path), exist_ok=True)
    with open(token_path, "w") as f:
        json.dump(tokens, f, indent=2)
    logger.info("Tokens saved to %s", token_path)
    logger.info("New access token: %s", tokens["access_token"])
    logger.info("New refresh token: %s", tokens["refresh_token"])
    logger.info("Expires in (seconds): %s", tokens["expires_in_seconds"])
    return tokens
