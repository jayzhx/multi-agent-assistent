import base64
import hashlib
import hmac
import os
import uuid
from typing import Any, Dict, Optional

from itsdangerous import BadSignature, URLSafeSerializer
from psycopg import Connection, IntegrityError

from customer_support_chat.app.core.database import get_connection as get_database_connection

secret_key = os.environ.get("WEB_APP_SECRET_KEY", "dev-web-app-secret")
serializer = URLSafeSerializer(secret_key, salt="travel-auth")


def get_connection() -> Connection:
    return get_database_connection(rows_as_dict=True)


def row_to_user(row: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if row is None:
        return None

    created_at = row["created_at"]
    return {
        "id": row["id"],
        "username": row["username"],
        "passenger_id": row["passenger_id"],
        "created_at": created_at.isoformat() if hasattr(created_at, "isoformat") else created_at,
    }


def hash_password(password: str) -> str:
    salt = os.urandom(16)
    password_hash = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 120000)
    return f"{base64.b64encode(salt).decode('utf-8')}${base64.b64encode(password_hash).decode('utf-8')}"


def verify_password(password: str, stored_password: str) -> bool:
    try:
        salt_text, hash_text = stored_password.split("$", maxsplit=1)
        salt = base64.b64decode(salt_text.encode("utf-8"))
        expected_hash = base64.b64decode(hash_text.encode("utf-8"))
    except (ValueError, TypeError):
        return False

    current_hash = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 120000)
    return hmac.compare_digest(current_hash, expected_hash)


def generate_passenger_id(cursor) -> str:
    while True:
        passenger_id = f"P-{uuid.uuid4().hex[:12].upper()}"
        cursor.execute("SELECT 1 FROM users WHERE passenger_id = %s", (passenger_id,))
        if cursor.fetchone() is None:
            return passenger_id


def initialize_auth_schema():
    conn = get_connection()
    cursor = conn.cursor()

    # 创建用户表，用于登录后识别具体用户。
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id BIGSERIAL PRIMARY KEY,
            username TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            passenger_id TEXT NOT NULL UNIQUE,
            created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    conn.commit()
    conn.close()


def get_user_by_id(user_id: int) -> Optional[Dict[str, Any]]:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, username, passenger_id, created_at FROM users WHERE id = %s", (user_id,))
    user = row_to_user(cursor.fetchone())
    conn.close()
    return user


def get_user_by_username(username: str) -> Optional[Dict[str, Any]]:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, username, passenger_id, created_at FROM users WHERE username = %s",
        (username,),
    )
    user = row_to_user(cursor.fetchone())
    conn.close()
    return user


def create_user(username: str, password: str, passenger_id: Optional[str] = None) -> Dict[str, Any]:
    normalized_username = username.strip()
    normalized_passenger_id = passenger_id.strip() if passenger_id else ""

    if not normalized_username:
        raise ValueError("用户名不能为空。")
    if len(password) < 6:
        raise ValueError("密码长度不能少于 6 位。")

    conn = get_connection()
    cursor = conn.cursor()

    final_passenger_id = normalized_passenger_id or generate_passenger_id(cursor)

    try:
        cursor.execute(
            """
            INSERT INTO users (username, password_hash, passenger_id)
            VALUES (%s, %s, %s)
            RETURNING id
            """,
            (
                normalized_username,
                hash_password(password),
                final_passenger_id,
            ),
        )
        user_id = cursor.fetchone()["id"]
        conn.commit()
    except IntegrityError as error:
        conn.rollback()
        conn.close()
        constraint_name = error.diag.constraint_name
        if constraint_name == "users_username_key":
            raise ValueError("该用户名已存在。")
        if constraint_name == "users_passenger_id_key":
            raise ValueError("该乘机人 ID 已被绑定，请更换后重试。")
        raise ValueError("注册失败，请检查输入信息。")

    conn.close()
    user = get_user_by_id(user_id)
    if user is None:
        raise ValueError("注册成功，但读取用户信息失败。")
    return user


def authenticate_user(username: str, password: str) -> Optional[Dict[str, Any]]:
    normalized_username = username.strip()
    if not normalized_username or not password:
        return None

    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT id, username, passenger_id, created_at, password_hash
        FROM users
        WHERE username = %s
        """,
        (normalized_username,),
    )
    row = cursor.fetchone()
    conn.close()

    if row is None:
        return None
    if not verify_password(password, row["password_hash"]):
        return None

    return row_to_user(row)


def create_auth_token(user_id: int) -> str:
    return serializer.dumps({"user_id": user_id})


def get_user_from_token(token: Optional[str]) -> Optional[Dict[str, Any]]:
    if not token:
        return None

    try:
        payload = serializer.loads(token)
    except BadSignature:
        return None

    user_id = payload.get("user_id")
    if not isinstance(user_id, int):
        return None

    return get_user_by_id(user_id)
