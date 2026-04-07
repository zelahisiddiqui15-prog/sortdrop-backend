import sqlite3
import uuid
from datetime import datetime

DB_PATH = "sortdrop.db"
TRIAL_LIMIT = 25


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            email TEXT UNIQUE,
            created_at TEXT,
            sorts_used INTEGER DEFAULT 0,
            trial_limit INTEGER DEFAULT 25,
            subscription_active INTEGER DEFAULT 0,
            stripe_customer_id TEXT,
            stripe_subscription_id TEXT
        )
    """)
    conn.commit()
    conn.close()


def create_user(email=None):
    conn = get_db()
    user_id = str(uuid.uuid4())
    now = datetime.utcnow().isoformat()
    conn.execute(
        "INSERT INTO users (id, email, created_at) VALUES (?, ?, ?)",
        (user_id, email, now)
    )
    conn.commit()
    conn.close()
    return user_id


def get_user(user_id):
    conn = get_db()
    user = conn.execute(
        "SELECT * FROM users WHERE id = ?", (user_id,)
    ).fetchone()
    conn.close()
    return dict(user) if user else None


def increment_sorts(user_id):
    conn = get_db()
    conn.execute(
        "UPDATE users SET sorts_used = sorts_used + 1 WHERE id = ?",
        (user_id,)
    )
    conn.commit()
    conn.close()


def activate_subscription(stripe_customer_id, stripe_subscription_id):
    conn = get_db()
    conn.execute(
        """UPDATE users SET subscription_active = 1,
           stripe_customer_id = ?, stripe_subscription_id = ?
           WHERE stripe_customer_id = ?""",
        (stripe_customer_id, stripe_subscription_id, stripe_customer_id)
    )
    conn.commit()
    conn.close()


def deactivate_subscription(stripe_customer_id):
    conn = get_db()
    conn.execute(
        "UPDATE users SET subscription_active = 0 WHERE stripe_customer_id = ?",
        (stripe_customer_id,)
    )
    conn.commit()
    conn.close()


def set_stripe_customer(user_id, stripe_customer_id):
    conn = get_db()
    conn.execute(
        "UPDATE users SET stripe_customer_id = ? WHERE id = ?",
        (stripe_customer_id, user_id)
    )
    conn.commit()
    conn.close()