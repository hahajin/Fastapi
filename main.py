# backend/main.py  — FastAPI version
import sys
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from server.database import Base, engine, SessionLocal
from server.user import User
from server.auth import get_password_hash

from server.routers import login, core, projects, admin
from routes.shell_routes import router as shell_router
from routes.ollama_routes import router as ollama_router
from routes.analysis_routes import router as analysis_router


# ─── Logger (optional – keeps parity with the Flask version) ────────────────

class Logger:
    def __init__(self, filename: str):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding="utf-8")

    def write(self, message: str):
        self.terminal.write(message)
        try:
            self.log.write(message)
            self.log.flush()
        except Exception as e:
            self.terminal.write(f"Fail to write to log: {e}\n")

    def flush(self):
        self.log.flush()

    def isatty(self):
        # 返回原始 terminal 的 isatty 状态，这样 uvicorn 能正确判断
        return self.terminal.isatty()


# ─── Startup / shutdown ──────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Create all tables on startup
    Base.metadata.create_all(bind=engine)

    # Ensure a default admin exists
    db = SessionLocal()
    try:
        admin_email    = os.getenv("DEFAULT_ADMIN_EMAIL",    "admin@example.com")
        admin_password = os.getenv("DEFAULT_ADMIN_PASSWORD", "admin_password")
        admin_username = os.getenv("DEFAULT_ADMIN_USERNAME", "admin")

        from sqlalchemy import select
        existing_admin = db.execute(
            select(User).where(User.is_admin == True)  # noqa: E712
        ).scalars().first()

        if not existing_admin:
            existing_user = db.execute(
                select(User).where(User.email == admin_email)
            ).scalars().first()

            if not existing_user:
                admin_user = User(
                    username=admin_username,
                    email=admin_email,
                    password_hash=get_password_hash(admin_password),
                    is_admin=True,
                )
                db.add(admin_user)
                db.commit()
                print(f"管理员账户已创建: {admin_email}")
            else:
                existing_user.is_admin = True
                db.commit()
                print(f"已存在的用户被设为管理员: {admin_email}")
    finally:
        db.close()

    yield  # application runs here


# ─── App factory ─────────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    app = FastAPI(
        title="FEA Platform API",
        description="API for the FEA Platform backend, built with FastAPI.",
        version="2.0.0",
        lifespan=lifespan,
    )

    # ── CORS ──────────────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173"],
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["Content-Type", "Authorization"],
        max_age=600,
    )

    # ── Routers  (url_prefix parity with Flask blueprints) ───────────────────
    app.include_router(login.router,     prefix="/api")
    app.include_router(core.router,     prefix="/core")
    app.include_router(projects.router, prefix="/api/projects")
    app.include_router(admin.router,    prefix="/api/admin")
    app.include_router(shell_router,    prefix="/core/shells")
    app.include_router(ollama_router,   prefix="/chat")
    app.include_router(analysis_router, prefix="/analysis")

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    sys.stdout = Logger("output.log")
    print("Process started, logging to output.log")
    uvicorn.run("main:app", host="0.0.0.0", port=5000, reload=True, use_colors=False)
