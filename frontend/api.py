import os
import requests
import streamlit as st

API = os.getenv("API_URL", "http://localhost:8000")


def err(r) -> str:
    try:
        d = r.json().get("detail")
    except Exception:  # noqa: BLE001
        d = None
    return d if isinstance(d, str) else "Invalid input (check the email format; password must be 8-72 characters)."


def call(method: str, path: str, **kw):
    """One place for every backend request: adds the login token, handles expiry and a down server."""
    headers = {"Authorization": f"Bearer {st.session_state['token']}"} if "token" in st.session_state else {}
    timeout = kw.pop("timeout", 30)
    try:
        r = requests.request(method, f"{API}{path}", headers=headers, timeout=timeout, **kw)
    except requests.RequestException:
        st.error("Backend not reachable. Is uvicorn running?")
        st.stop()
    if r.status_code == 401 and "token" in st.session_state:   # expired / invalid token
        st.session_state.clear()
        st.session_state["notice"] = "Session expired. Please log in again."
        st.rerun()
    return r
