import json
import os
import secrets
import sqlite3
import string
import time
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError, InvalidHashError
from fastapi import FastAPI, Request, Response, HTTPException, Depends, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from itsdangerous import URLSafeTimedSerializer, BadSignature, SignatureExpired
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "app.db"
UPLOADS_DIR = BASE_DIR / "uploads"
UPLOADS_DIR.mkdir(exist_ok=True)

SUPERADMIN_USER = os.environ["SUPERADMIN_USER"]
SUPERADMIN_PASSWORD_HASH = os.environ["SUPERADMIN_PASSWORD_HASH"]
SESSION_SECRET = os.environ["SESSION_SECRET"]

SESSION_COOKIE = "session"
SESSION_MAX_AGE = 7 * 24 * 3600  # 7 giorni

ph = PasswordHasher()
serializer = URLSafeTimedSerializer(SESSION_SECRET, salt="segnalazioni-session")

CATEGORIES = {"buca", "illuminazione", "segnaletica", "verde", "rifiuti", "arredo", "allagamento", "altro"}
STATUSES = {"aperto", "in_lavorazione", "risolto", "rifiutato"}
PRIORITIES = {"bassa", "media", "alta"}

# Bounding box approssimativo di Castelfranco Veneto e dintorni.
LAT_MIN, LAT_MAX = 45.55, 45.80
LNG_MIN, LNG_MAX = 11.75, 12.10

MAX_PHOTO_BYTES = 8 * 1024 * 1024
ALLOWED_PHOTO_TYPES = {"image/jpeg": "jpg", "image/png": "png", "image/webp": "webp"}

# ─────────────────────────────────────────────────────────────
# Rate limit login: max 8 tentativi / 5 min per (ip, username)
# ─────────────────────────────────────────────────────────────
_login_attempts: dict[str, list[float]] = {}
LOGIN_WINDOW_SECONDS = 300
LOGIN_MAX_ATTEMPTS = 8


def _rate_limit_key(request: Request, username: str) -> str:
    return f"{request.client.host}:{username.lower()}"


def check_rate_limit(request: Request, username: str):
    key = _rate_limit_key(request, username)
    now = time.time()
    attempts = [t for t in _login_attempts.get(key, []) if now - t < LOGIN_WINDOW_SECONDS]
    _login_attempts[key] = attempts
    if len(attempts) >= LOGIN_MAX_ATTEMPTS:
        raise HTTPException(status_code=429, detail="Troppi tentativi, riprova più tardi")


def record_login_attempt(request: Request, username: str):
    key = _rate_limit_key(request, username)
    _login_attempts.setdefault(key, []).append(time.time())


# ─────────────────────────────────────────────────────────────
# DB
# ─────────────────────────────────────────────────────────────
@contextmanager
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with get_db() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
              id                   INTEGER PRIMARY KEY AUTOINCREMENT,
              username             TEXT UNIQUE NOT NULL,
              password_hash        TEXT NOT NULL,
              display_name         TEXT NOT NULL,
              role                 TEXT NOT NULL DEFAULT 'user',
              active               INTEGER NOT NULL DEFAULT 1,
              must_change_password INTEGER NOT NULL DEFAULT 1,
              created_at           TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS reports (
              id          INTEGER PRIMARY KEY AUTOINCREMENT,
              author_id   INTEGER NOT NULL,
              title       TEXT NOT NULL,
              category    TEXT NOT NULL,
              description TEXT NOT NULL,
              lat         REAL NOT NULL,
              lng         REAL NOT NULL,
              status      TEXT NOT NULL DEFAULT 'aperto',
              priority    TEXT,
              photo_path  TEXT,
              created_at  TEXT NOT NULL,
              updated_at  TEXT NOT NULL,
              FOREIGN KEY (author_id) REFERENCES users(id)
            );

            CREATE TABLE IF NOT EXISTS report_events (
              id         INTEGER PRIMARY KEY AUTOINCREMENT,
              report_id  INTEGER NOT NULL,
              actor_id   INTEGER NOT NULL,
              type       TEXT NOT NULL,
              payload    TEXT,
              created_at TEXT NOT NULL,
              FOREIGN KEY (report_id) REFERENCES reports(id)
            );
            """
        )


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ─────────────────────────────────────────────────────────────
# Sessioni
# ─────────────────────────────────────────────────────────────
class Session:
    def __init__(self, user_id: int, role: str, display_name: str):
        self.user_id = user_id
        self.role = role
        self.display_name = display_name


def create_session_cookie(response: Response, user_id: int, role: str, display_name: str):
    token = serializer.dumps({"uid": user_id, "role": role, "dn": display_name})
    response.set_cookie(
        SESSION_COOKIE,
        token,
        max_age=SESSION_MAX_AGE,
        httponly=True,
        secure=True,
        samesite="lax",
        path="/",
    )


def clear_session_cookie(response: Response):
    response.delete_cookie(SESSION_COOKIE, path="/")


def get_session(request: Request) -> Optional[Session]:
    token = request.cookies.get(SESSION_COOKIE)
    if not token:
        return None
    try:
        data = serializer.loads(token, max_age=SESSION_MAX_AGE)
    except (BadSignature, SignatureExpired):
        return None
    return Session(user_id=data["uid"], role=data["role"], display_name=data["dn"])


def require_session(request: Request) -> Session:
    """1) sessione valida? 2) se non superadmin, utente ancora active?"""
    session = get_session(request)
    if session is None:
        raise HTTPException(status_code=401, detail="Non autenticato")

    if session.role != "superadmin":
        with get_db() as conn:
            row = conn.execute(
                "SELECT active, role, display_name FROM users WHERE id = ?", (session.user_id,)
            ).fetchone()
        if row is None or not row["active"]:
            raise HTTPException(status_code=403, detail="Utente disattivato")
        # rispecchia eventuali cambi di ruolo fatti nel frattempo dall'admin
        session.role = row["role"]
        session.display_name = row["display_name"]

    return session


def require_moderator(session: Session = Depends(require_session)) -> Session:
    if session.role not in ("superadmin", "moderator"):
        raise HTTPException(status_code=403, detail="Permessi insufficienti")
    return session


def require_superadmin(session: Session = Depends(require_session)) -> Session:
    if session.role != "superadmin":
        raise HTTPException(status_code=403, detail="Permessi insufficienti")
    return session


# ─────────────────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────────────────
app = FastAPI()
init_db()


@app.middleware("http")
async def no_store_api(request: Request, call_next):
    response = await call_next(request)
    if request.url.path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-store"
    return response


# ─────────────────────────────────────────────────────────────
# Auth endpoints
# ─────────────────────────────────────────────────────────────
def generate_temp_password() -> str:
    alphabet = string.ascii_uppercase + string.ascii_lowercase + string.digits + "!@#$%&*"
    while True:
        pw = "".join(secrets.choice(alphabet) for _ in range(14))
        if (
            any(c.islower() for c in pw)
            and any(c.isupper() for c in pw)
            and any(c.isdigit() for c in pw)
            and any(c in "!@#$%&*" for c in pw)
        ):
            return pw


@app.post("/api/login")
async def login(request: Request, response: Response):
    body = await request.json()
    username = str(body.get("username", "")).strip()
    password = str(body.get("password", ""))

    if not username or not password:
        raise HTTPException(status_code=401, detail="Credenziali non valide")

    check_rate_limit(request, username)
    record_login_attempt(request, username)

    # 1. superadmin
    if secrets.compare_digest(username, SUPERADMIN_USER):
        try:
            ph.verify(SUPERADMIN_PASSWORD_HASH, password)
        except (VerifyMismatchError, InvalidHashError):
            raise HTTPException(status_code=401, detail="Credenziali non valide")
        create_session_cookie(response, user_id=0, role="superadmin", display_name="Superadmin")
        return {"user_id": 0, "display_name": "Superadmin", "role": "superadmin", "must_change_password": False}

    # 2. utente normale
    with get_db() as conn:
        row = conn.execute(
            "SELECT id, password_hash, display_name, role, active, must_change_password "
            "FROM users WHERE username = ?",
            (username,),
        ).fetchone()

    if row is None:
        raise HTTPException(status_code=401, detail="Credenziali non valide")

    try:
        ph.verify(row["password_hash"], password)
    except (VerifyMismatchError, InvalidHashError):
        raise HTTPException(status_code=401, detail="Credenziali non valide")

    if not row["active"]:
        raise HTTPException(status_code=401, detail="Credenziali non valide")

    create_session_cookie(response, user_id=row["id"], role=row["role"], display_name=row["display_name"])
    return {
        "user_id": row["id"],
        "display_name": row["display_name"],
        "role": row["role"],
        "must_change_password": bool(row["must_change_password"]),
    }


@app.post("/api/logout")
async def logout(response: Response):
    clear_session_cookie(response)
    return {"ok": True}


@app.get("/api/me")
async def me(request: Request):
    session = get_session(request)
    if session is None:
        return {"authenticated": False}

    if session.role == "superadmin":
        return {
            "authenticated": True,
            "user_id": 0,
            "display_name": "Superadmin",
            "role": "superadmin",
            "must_change_password": False,
        }

    with get_db() as conn:
        row = conn.execute(
            "SELECT id, display_name, role, active, must_change_password FROM users WHERE id = ?",
            (session.user_id,),
        ).fetchone()
    if row is None or not row["active"]:
        return {"authenticated": False}

    return {
        "authenticated": True,
        "user_id": row["id"],
        "display_name": row["display_name"],
        "role": row["role"],
        "must_change_password": bool(row["must_change_password"]),
    }


@app.post("/api/me/password")
async def change_password(request: Request, session: Session = Depends(require_session)):
    if session.role == "superadmin":
        raise HTTPException(status_code=400, detail="Password superadmin gestita via env")

    body = await request.json()
    new_password = str(body.get("new_password", ""))
    if len(new_password) < 10:
        raise HTTPException(status_code=400, detail="Password troppo corta (minimo 10 caratteri)")

    new_hash = ph.hash(new_password)
    with get_db() as conn:
        conn.execute(
            "UPDATE users SET password_hash = ?, must_change_password = 0 WHERE id = ?",
            (new_hash, session.user_id),
        )
    return {"ok": True}


# ─────────────────────────────────────────────────────────────
# Reports
# ─────────────────────────────────────────────────────────────
def row_to_report(row: sqlite3.Row) -> dict:
    return {
        "id": row["id"],
        "author_id": row["author_id"],
        "title": row["title"],
        "category": row["category"],
        "description": row["description"],
        "lat": row["lat"],
        "lng": row["lng"],
        "status": row["status"],
        "priority": row["priority"],
        "photo_path": f"/uploads/{row['photo_path']}" if row["photo_path"] else None,
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


@app.get("/api/reports")
async def list_reports(
    request: Request,
    status: Optional[str] = None,
    category: Optional[str] = None,
    session: Session = Depends(require_session),
):
    query = "SELECT * FROM reports WHERE 1=1"
    params: list = []
    if status:
        if status not in STATUSES:
            raise HTTPException(status_code=400, detail="status non valido")
        query += " AND status = ?"
        params.append(status)
    if category:
        if category not in CATEGORIES:
            raise HTTPException(status_code=400, detail="category non valida")
        query += " AND category = ?"
        params.append(category)
    query += " ORDER BY created_at DESC"

    with get_db() as conn:
        rows = conn.execute(query, params).fetchall()
    return [row_to_report(r) for r in rows]


def _validate_report_fields(title: str, category: str, description: str, lat: float, lng: float,
                             priority: Optional[str]):
    if not title or not title.strip():
        raise HTTPException(status_code=400, detail="title obbligatorio")
    if len(title) > 200:
        raise HTTPException(status_code=400, detail="title troppo lungo")
    if category not in CATEGORIES:
        raise HTTPException(status_code=400, detail="category non valida")
    if not description or not description.strip():
        raise HTTPException(status_code=400, detail="description obbligatoria")
    if len(description) > 5000:
        raise HTTPException(status_code=400, detail="description troppo lunga")
    if not (LAT_MIN <= lat <= LAT_MAX and LNG_MIN <= lng <= LNG_MAX):
        raise HTTPException(status_code=400, detail="coordinate fuori dall'area consentita")
    if priority is not None and priority not in PRIORITIES:
        raise HTTPException(status_code=400, detail="priority non valida")


def strip_exif_and_save(upload_bytes: bytes, content_type: str) -> str:
    if content_type not in ALLOWED_PHOTO_TYPES:
        raise HTTPException(status_code=400, detail="Formato immagine non supportato")
    if len(upload_bytes) > MAX_PHOTO_BYTES:
        raise HTTPException(status_code=400, detail="Immagine troppo grande")

    from io import BytesIO

    try:
        img = Image.open(BytesIO(upload_bytes))
        img.verify()
        img = Image.open(BytesIO(upload_bytes))  # riapri dopo verify()
    except Exception:
        raise HTTPException(status_code=400, detail="Immagine non valida")

    # ricrea l'immagine senza metadati EXIF (niente GPS)
    clean = Image.new(img.mode if img.mode in ("RGB", "L") else "RGB", img.size)
    clean.paste(img.convert(clean.mode))

    ext = ALLOWED_PHOTO_TYPES[content_type]
    filename = f"{secrets.token_hex(16)}.{ext}"
    dest = UPLOADS_DIR / filename
    save_format = "JPEG" if ext == "jpg" else ext.upper()
    clean.save(dest, format=save_format)
    return filename


@app.post("/api/reports")
async def create_report(
    request: Request,
    title: str = Form(...),
    category: str = Form(...),
    description: str = Form(...),
    lat: float = Form(...),
    lng: float = Form(...),
    priority: Optional[str] = Form(None),
    photo: Optional[UploadFile] = File(None),
    session: Session = Depends(require_session),
):
    if session.role == "superadmin":
        raise HTTPException(status_code=400, detail="Il superadmin non crea segnalazioni")

    _validate_report_fields(title, category, description, lat, lng, priority)

    photo_filename = None
    if photo is not None and photo.filename:
        content = await photo.read()
        photo_filename = strip_exif_and_save(content, photo.content_type or "")

    ts = now_iso()
    with get_db() as conn:
        cur = conn.execute(
            "INSERT INTO reports (author_id, title, category, description, lat, lng, status, priority, "
            "photo_path, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, 'aperto', ?, ?, ?, ?)",
            (session.user_id, title.strip(), category, description.strip(), lat, lng, priority,
             photo_filename, ts, ts),
        )
        report_id = cur.lastrowid
        conn.execute(
            "INSERT INTO report_events (report_id, actor_id, type, payload, created_at) VALUES (?, ?, 'edit', ?, ?)",
            (report_id, session.user_id, json.dumps({"action": "created"}), ts),
        )
        row = conn.execute("SELECT * FROM reports WHERE id = ?", (report_id,)).fetchone()
    return row_to_report(row)


@app.get("/api/reports/{report_id}")
async def get_report(report_id: int, session: Session = Depends(require_session)):
    with get_db() as conn:
        row = conn.execute("SELECT * FROM reports WHERE id = ?", (report_id,)).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="Non trovato")
        events = conn.execute(
            "SELECT * FROM report_events WHERE report_id = ? ORDER BY created_at ASC", (report_id,)
        ).fetchall()

    report = row_to_report(row)
    report["events"] = [
        {
            "id": e["id"],
            "actor_id": e["actor_id"],
            "type": e["type"],
            "payload": json.loads(e["payload"]) if e["payload"] else None,
            "created_at": e["created_at"],
        }
        for e in events
    ]
    return report


def _can_write_report(session: Session, author_id: int) -> bool:
    return session.role in ("superadmin", "moderator") or session.user_id == author_id


@app.patch("/api/reports/{report_id}")
async def update_report(report_id: int, request: Request, session: Session = Depends(require_session)):
    with get_db() as conn:
        row = conn.execute("SELECT * FROM reports WHERE id = ?", (report_id,)).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="Non trovato")
        if not _can_write_report(session, row["author_id"]):
            raise HTTPException(status_code=403, detail="Permessi insufficienti")

        body = await request.json()
        updates = {}
        if "title" in body:
            title = str(body["title"])
            if not title.strip() or len(title) > 200:
                raise HTTPException(status_code=400, detail="title non valido")
            updates["title"] = title.strip()
        if "category" in body:
            if body["category"] not in CATEGORIES:
                raise HTTPException(status_code=400, detail="category non valida")
            updates["category"] = body["category"]
        if "description" in body:
            desc = str(body["description"])
            if not desc.strip() or len(desc) > 5000:
                raise HTTPException(status_code=400, detail="description non valida")
            updates["description"] = desc.strip()
        if "priority" in body:
            pr = body["priority"]
            if pr is not None and pr not in PRIORITIES:
                raise HTTPException(status_code=400, detail="priority non valida")
            updates["priority"] = pr

        if not updates:
            return row_to_report(row)

        ts = now_iso()
        set_clause = ", ".join(f"{k} = ?" for k in updates) + ", updated_at = ?"
        params = list(updates.values()) + [ts, report_id]
        conn.execute(f"UPDATE reports SET {set_clause} WHERE id = ?", params)
        conn.execute(
            "INSERT INTO report_events (report_id, actor_id, type, payload, created_at) VALUES (?, ?, 'edit', ?, ?)",
            (report_id, session.user_id, json.dumps({"fields": list(updates.keys())}), ts),
        )
        updated = conn.execute("SELECT * FROM reports WHERE id = ?", (report_id,)).fetchone()
    return row_to_report(updated)


@app.patch("/api/reports/{report_id}/status")
async def update_report_status(
    report_id: int, request: Request, session: Session = Depends(require_moderator)
):
    body = await request.json()
    new_status = body.get("status")
    if new_status not in STATUSES:
        raise HTTPException(status_code=400, detail="status non valido")

    with get_db() as conn:
        row = conn.execute("SELECT * FROM reports WHERE id = ?", (report_id,)).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="Non trovato")

        ts = now_iso()
        conn.execute(
            "UPDATE reports SET status = ?, updated_at = ? WHERE id = ?", (new_status, ts, report_id)
        )
        actor_id = 0 if session.role == "superadmin" else session.user_id
        conn.execute(
            "INSERT INTO report_events (report_id, actor_id, type, payload, created_at) VALUES (?, ?, "
            "'status_change', ?, ?)",
            (report_id, actor_id, json.dumps({"from": row["status"], "to": new_status}), ts),
        )
        updated = conn.execute("SELECT * FROM reports WHERE id = ?", (report_id,)).fetchone()
    return row_to_report(updated)


@app.delete("/api/reports/{report_id}")
async def delete_report(report_id: int, session: Session = Depends(require_session)):
    with get_db() as conn:
        row = conn.execute("SELECT * FROM reports WHERE id = ?", (report_id,)).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="Non trovato")

        is_owner_deletable = session.user_id == row["author_id"] and row["status"] == "aperto"
        if not (is_owner_deletable or session.role == "superadmin"):
            raise HTTPException(status_code=403, detail="Permessi insufficienti")

        if row["photo_path"]:
            photo_file = UPLOADS_DIR / row["photo_path"]
            if photo_file.exists():
                photo_file.unlink()

        conn.execute("DELETE FROM report_events WHERE report_id = ?", (report_id,))
        conn.execute("DELETE FROM reports WHERE id = ?", (report_id,))
    return {"ok": True}


# ─────────────────────────────────────────────────────────────
# Admin utenti (solo superadmin)
# ─────────────────────────────────────────────────────────────
@app.get("/api/admin/users")
async def list_users(session: Session = Depends(require_superadmin)):
    with get_db() as conn:
        rows = conn.execute(
            "SELECT id, username, display_name, role, active, must_change_password, created_at "
            "FROM users ORDER BY created_at DESC"
        ).fetchall()
    return [dict(r) for r in rows]


@app.post("/api/admin/users")
async def create_user(request: Request, session: Session = Depends(require_superadmin)):
    body = await request.json()
    username = str(body.get("username", "")).strip().lower()
    display_name = str(body.get("display_name", "")).strip()
    role = body.get("role", "user")

    if not username or not display_name:
        raise HTTPException(status_code=400, detail="username e display_name obbligatori")
    if role not in ("user", "moderator"):
        raise HTTPException(status_code=400, detail="role non valido")
    if secrets.compare_digest(username, SUPERADMIN_USER.lower()):
        raise HTTPException(status_code=400, detail="username riservato")

    temp_password = generate_temp_password()
    password_hash = ph.hash(temp_password)

    with get_db() as conn:
        existing = conn.execute("SELECT id FROM users WHERE username = ?", (username,)).fetchone()
        if existing:
            raise HTTPException(status_code=409, detail="username già esistente")
        cur = conn.execute(
            "INSERT INTO users (username, password_hash, display_name, role, active, must_change_password, "
            "created_at) VALUES (?, ?, ?, ?, 1, 1, ?)",
            (username, password_hash, display_name, role, now_iso()),
        )
        user_id = cur.lastrowid

    return {"id": user_id, "username": username, "temp_password": temp_password}


@app.patch("/api/admin/users/{user_id}")
async def update_user(user_id: int, request: Request, session: Session = Depends(require_superadmin)):
    body = await request.json()
    updates = {}
    if "active" in body:
        updates["active"] = 1 if body["active"] else 0
    if "display_name" in body:
        dn = str(body["display_name"]).strip()
        if not dn:
            raise HTTPException(status_code=400, detail="display_name non valido")
        updates["display_name"] = dn
    if "role" in body:
        if body["role"] not in ("user", "moderator"):
            raise HTTPException(status_code=400, detail="role non valido")
        updates["role"] = body["role"]

    if not updates:
        raise HTTPException(status_code=400, detail="nessun campo da aggiornare")

    with get_db() as conn:
        existing = conn.execute("SELECT id FROM users WHERE id = ?", (user_id,)).fetchone()
        if existing is None:
            raise HTTPException(status_code=404, detail="Non trovato")
        set_clause = ", ".join(f"{k} = ?" for k in updates)
        conn.execute(f"UPDATE users SET {set_clause} WHERE id = ?", list(updates.values()) + [user_id])
        row = conn.execute(
            "SELECT id, username, display_name, role, active, must_change_password, created_at "
            "FROM users WHERE id = ?",
            (user_id,),
        ).fetchone()
    return dict(row)


@app.post("/api/admin/users/{user_id}/reset-password")
async def reset_password(user_id: int, session: Session = Depends(require_superadmin)):
    temp_password = generate_temp_password()
    password_hash = ph.hash(temp_password)
    with get_db() as conn:
        existing = conn.execute("SELECT username FROM users WHERE id = ?", (user_id,)).fetchone()
        if existing is None:
            raise HTTPException(status_code=404, detail="Non trovato")
        conn.execute(
            "UPDATE users SET password_hash = ?, must_change_password = 1 WHERE id = ?",
            (password_hash, user_id),
        )
    return {"id": user_id, "username": existing["username"], "temp_password": temp_password}


# ─────────────────────────────────────────────────────────────
# Static files: foto (solo autenticati) + app
# ─────────────────────────────────────────────────────────────
@app.get("/uploads/{filename}")
async def get_upload(filename: str, session: Session = Depends(require_session)):
    if "/" in filename or "\\" in filename or ".." in filename:
        raise HTTPException(status_code=400, detail="nome file non valido")
    path = UPLOADS_DIR / filename
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Non trovato")
    return FileResponse(path)


@app.get("/")
async def index():
    return FileResponse(BASE_DIR / "index.html")


app.mount("/", StaticFiles(directory=str(BASE_DIR), html=True), name="static")
