from datetime import datetime, timedelta, timezone
import bcrypt
import jwt
from .config import get_settings


def hash_password(password: str) -> str:
    # bcrypt salts automatically and is deliberately slow (resists brute force)
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), hashed.encode("utf-8"))
    except ValueError:
        return False


def _secret() -> str:
    s = get_settings().jwt_secret
    if len(s) < 32:
        raise RuntimeError(
            "JWT_SECRET is missing or shorter than 32 chars. Generate one with: "
            'python -c "import secrets; print(secrets.token_hex(32))" and add JWT_SECRET=... to backend/.env')
    return s


def create_access_token(user_id: int) -> str:
    st = get_settings()
    now = datetime.now(timezone.utc)
    payload = {"sub": str(user_id), "iat": now,
               "exp": now + timedelta(minutes=st.jwt_expire_minutes)}
    return jwt.encode(payload, _secret(), algorithm=st.jwt_algorithm)


def decode_token(token: str) -> int | None:
    """Returns the user id, or None if the token is invalid/expired/tampered."""
    try:
        data = jwt.decode(token, _secret(), algorithms=[get_settings().jwt_algorithm])
        return int(data["sub"])
    except (jwt.PyJWTError, KeyError, ValueError):
        return None
