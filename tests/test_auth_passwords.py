import hmac
import os

import app
import db


def test_register_stores_salted_hash(temp_db):
    password = "S3cur3!Pass"
    body = app.RegisterBody(user_id="new_user", email="new@example.com", password=password)
    app.auth_register(body)

    row = db.get_user_auth("new_user")
    assert row is not None
    assert row["pw_salt"]
    assert len(row["pw_salt"]) == 32
    # Hash should not match the deterministic legacy hash format.
    legacy_hash = hmac.new(
        os.getenv("AUTH_SALT", "local_salt").encode("utf-8"),
        password.encode("utf-8"),
        digestmod="sha256",
    ).hexdigest()
    assert row["pw_hash"] != legacy_hash


def test_login_verifies_pbkdf2_password(temp_db):
    password = "Another#Pass1"
    pw_hash, pw_salt = app._hash_password(password)
    db.create_user("pbkdf2_user", "pb@example.com", pw_hash, pw_salt)

    result = app.auth_login(app.LoginBody(user_id="pbkdf2_user", password=password))
    assert result["user_id"] == "pbkdf2_user"
    assert result["token"]


def test_login_upgrades_legacy_password(temp_db, monkeypatch):
    user_id = "legacy_user"
    password = "LegacyPass!"
    monkeypatch.setenv("AUTH_SALT", "legacy_salt")
    legacy_hash = hmac.new(
        b"legacy_salt",
        password.encode("utf-8"),
        digestmod="sha256",
    ).hexdigest()
    db.create_user(user_id, "legacy@example.com", legacy_hash, None)

    result = app.auth_login(app.LoginBody(user_id=user_id, password=password))
    assert result["user_id"] == user_id

    upgraded = db.get_user_auth(user_id)
    assert upgraded["pw_salt"]
    assert upgraded["pw_hash"] != legacy_hash
