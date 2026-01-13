#!/usr/bin/env python
"""
FYERS Access Token Generator
Run: python src/data/generate_token.py
"""

import os
import webbrowser
from urllib.parse import urlparse, parse_qs

from fyers_apiv3 import fyersModel
from dotenv import load_dotenv

load_dotenv()

CLIENT_ID = os.getenv("FYERS_CLIENT_ID")
SECRET_KEY = os.getenv("FYERS_SECRET_KEY")
REDIRECT_URI = os.getenv("FYERS_REDIRECT_URI", "https://google.com")


def create_session() -> fyersModel.SessionModel:
    return fyersModel.SessionModel(
        client_id=CLIENT_ID,
        secret_key=SECRET_KEY,
        redirect_uri=REDIRECT_URI,
        response_type="code",
        grant_type="authorization_code"
    )


def extract_auth_code(url: str) -> str:
    parsed = urlparse(url)
    params = parse_qs(parsed.query)
    return params.get("auth_code", params.get("code", [None]))[0]


def update_env(token: str) -> bool:
    env_path = ".env"
    lines = []

    if os.path.exists(env_path):
        with open(env_path, "r") as f:
            lines = f.readlines()

    found = False
    for i, line in enumerate(lines):
        if line.startswith("FYERS_ACCESS_TOKEN"):
            lines[i] = f"FYERS_ACCESS_TOKEN={token}\n"
            found = True
            break

    if not found:
        lines.append(f"FYERS_ACCESS_TOKEN={token}\n")

    with open(env_path, "w") as f:
        f.writelines(lines)

    return True


def main():
    if not CLIENT_ID or not SECRET_KEY:
        print("Error: Configure FYERS_CLIENT_ID and FYERS_SECRET_KEY in .env")
        return

    session = create_session()
    auth_url = session.generate_authcode()

    print(f"\nOpening browser for FYERS login...")
    print(f"URL: {auth_url}\n")
    webbrowser.open(auth_url)

    redirect_url = input("Paste the redirect URL after login: ").strip()
    auth_code = extract_auth_code(redirect_url)

    if not auth_code:
        print("Error: Could not extract auth_code from URL")
        return

    session.set_token(auth_code)
    response = session.generate_token()

    if response.get("code") != 200:
        print(f"Error: {response.get('message', 'Token generation failed')}")
        return

    token = response["access_token"]
    update_env(token)
    print(f"\nAccess token saved to .env")


if __name__ == "__main__":
    main()
