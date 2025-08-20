from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from typing import Generator
from app.config.settings import settings
import structlog

logger = structlog.get_logger()

from sqlalchemy.engine import make_url
from pathlib import Path
import tempfile


# Build engine lazily later after resolving potential fallbacks
def _create_engine_from_url(db_url: str):
    connect_args = {"check_same_thread": False} if "sqlite" in db_url else {}
    return create_engine(db_url, connect_args=connect_args)


# Attempt to ensure sqlite parent dir exists and fall back to a temp file if needed.
def _resolve_database_url(original_url: str) -> str:
    try:
        url = make_url(original_url)
        if url.drivername and url.drivername.startswith("sqlite") and url.database:
            db_name = url.database
            db_path = Path(db_name)
            if not db_path.is_absolute():
                db_path = (Path.cwd() / db_path).resolve()

            db_dir = db_path.parent
            # Log resolved path for diagnostics
            logger.info("Resolved sqlite path", resolved=str(db_path), original=original_url)

            try:
                db_dir.mkdir(parents=True, exist_ok=True)
                # test writability by creating a temp file
                test_file = db_dir / ".writable_test"
                with open(test_file, "w") as f:
                    f.write("ok")
                test_file.unlink()
                return original_url
            except Exception as e:
                logger.error("Configured sqlite path not writable; falling back to temp file", error=str(e), path=str(db_path))
                tmp = Path(tempfile.gettempdir()) / "testcases_fallback.db"
                fallback = f"sqlite:///{tmp.as_posix()}"
                logger.info("Using fallback sqlite path", fallback=fallback)
                return fallback
    except Exception as e:
        logger.debug("Failed to parse/resolve database url", error=str(e), original=original_url)

    return original_url


# Resolve database url and create engine
resolved_db_url = _resolve_database_url(settings.database_url)
engine = _create_engine_from_url(resolved_db_url)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def get_database() -> Generator[Session, None, None]:
    """Dependency to get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def create_tables():
    """Create all database tables"""
    try:
        # Nothing to do here; engine initialization already resolved and logged any fallbacks.
        from app.models.database import Base

        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error("Failed to create database tables", error=str(e))
        raise
