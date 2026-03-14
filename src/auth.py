import os
import sqlite3
import hashlib
import secrets
from typing import Optional, Dict


# =========================
# PATHS
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
DB_PATH = os.path.join(DATA_DIR, "users.db")


# =========================
# DATABASE SETUP
# =========================
def ensure_db() -> None:
    """
    Create the SQLite database and users table if they do not already exist.
    """
    os.makedirs(DATA_DIR, exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            full_name TEXT NOT NULL,
            email TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            salt TEXT NOT NULL,
            role TEXT NOT NULL CHECK(role IN ('doctor', 'nurse', 'public')),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    conn.close()


# =========================
# PASSWORD SECURITY
# =========================
def hash_password(password: str, salt: str) -> str:
    """
    Hash a password with PBKDF2-HMAC-SHA256.
    """
    hashed = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        100_000
    )
    return hashed.hex()


# =========================
# USER MANAGEMENT
# =========================
def create_user(full_name: str, email: str, password: str, role: str) -> Dict:
    """
    Create a new user account.
    Returns a dict with success flag and message.
    """
    ensure_db()

    full_name = full_name.strip()
    email = email.strip().lower()
    role = role.strip().lower()

    if role not in {"doctor", "nurse", "public"}:
        return {"success": False, "message": "Invalid role."}

    if len(full_name) < 3:
        return {"success": False, "message": "Full name must contain at least 3 characters."}

    if "@" not in email or "." not in email:
        return {"success": False, "message": "Invalid email address."}

    if len(password) < 6:
        return {"success": False, "message": "Password must contain at least 6 characters."}

    salt = secrets.token_hex(16)
    password_hash = hash_password(password, salt)

    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO users (full_name, email, password_hash, salt, role)
            VALUES (?, ?, ?, ?, ?)
        """, (full_name, email, password_hash, salt, role))

        conn.commit()
        conn.close()

        return {"success": True, "message": "Account created successfully."}

    except sqlite3.IntegrityError:
        return {"success": False, "message": "An account with this email already exists."}


def authenticate_user(email: str, password: str) -> Optional[Dict]:
    """
    Authenticate a user by email and password.
    Returns user info dict if valid, otherwise None.
    """
    ensure_db()

    email = email.strip().lower()

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT id, full_name, email, password_hash, salt, role
        FROM users
        WHERE email = ?
    """, (email,))

    row = cursor.fetchone()
    conn.close()

    if row is None:
        return None

    user_id, full_name, email_db, password_hash_db, salt, role = row
    entered_hash = hash_password(password, salt)

    if entered_hash != password_hash_db:
        return None

    return {
        "id": user_id,
        "full_name": full_name,
        "email": email_db,
        "role": role
    }


def get_user_by_email(email: str) -> Optional[Dict]:
    """
    Fetch user metadata by email.
    """
    ensure_db()

    email = email.strip().lower()

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT id, full_name, email, role, created_at
        FROM users
        WHERE email = ?
    """, (email,))

    row = cursor.fetchone()
    conn.close()

    if row is None:
        return None

    user_id, full_name, email_db, role, created_at = row

    return {
        "id": user_id,
        "full_name": full_name,
        "email": email_db,
        "role": role,
        "created_at": created_at
    }