"""
Cleanup utility for the backend data stores.

Features:
 - Back up the SQLite DB file and Chroma persist directory.
 - Mode "vectors" (default):
     * NULL out embedding_vector in SQLite for all test cases.
     * Delete the configured Chroma collection (clears all vectors).
 - Mode "all":
     * Delete all rows from test_cases.
     * Delete the configured Chroma collection.
 - VACUUM the SQLite DB after modifications.

Usage examples:
  python scripts/cleanup_db.py --mode vectors --yes
  python scripts/cleanup_db.py --mode all --yes

Notes:
 - Uses app.config.settings for DATABASE_URL and Chroma paths.
 - Creates backups under ./data/backups/ with a timestamp.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Optional

import chromadb
from sqlalchemy import text

# Ensure we can import the app package when running as a script
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.config.settings import settings  # noqa: E402
from app.core.database import SessionLocal  # noqa: E402


def parse_sqlite_path(database_url: str) -> Optional[Path]:
    """Extract SQLite file path from a SQLAlchemy database URL.

    Supports formats like:
      - sqlite:///./data/testcases.db
      - sqlite:////absolute/path/to/testcases.db
    """
    prefix = "sqlite:///"
    if not database_url.startswith(prefix):
        return None
    raw_path = database_url[len(prefix):]
    # Normalize path separators on Windows
    raw_path = raw_path.replace("\\", "/")
    p = Path(raw_path)
    if not p.is_absolute():
        p = (REPO_ROOT / p).resolve()
    return p


def backup_sqlite(db_path: Path, backups_dir: Path) -> Optional[Path]:
    if not db_path.exists():
        return None
    backups_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    backup_path = backups_dir / f"testcases-{ts}.db"
    shutil.copy2(db_path, backup_path)
    return backup_path


def backup_chroma_dir(chroma_dir: Path, backups_dir: Path) -> Optional[Path]:
    if not chroma_dir.exists():
        return None
    backups_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    dest = backups_dir / f"chroma_db-{ts}"
    shutil.copytree(chroma_dir, dest)
    return dest


def wipe_chroma_collection() -> None:
    client = chromadb.PersistentClient(path=settings.chroma_persist_directory)
    try:
        client.delete_collection(settings.chroma_collection_name)
    except Exception:
        # If it doesn't exist, ignore
        pass


def cleanup_vectors(session) -> None:
    # Set embedding_vector to NULL for all rows
    session.execute(text("UPDATE test_cases SET embedding_vector = NULL"))
    session.commit()
    # VACUUM database (requires autocommit off, use connection)
    try:
        session.execute(text("VACUUM"))
    except Exception:
        # Some SQLite setups require executing VACUUM outside transaction; ignore if it fails
        pass


def cleanup_all(session) -> None:
    session.execute(text("DELETE FROM test_cases"))
    session.commit()
    try:
        session.execute(text("VACUUM"))
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(description="Clean up SQLite and Chroma data")
    parser.add_argument("--mode", choices=["vectors", "all"], default="vectors", help="Cleanup mode")
    parser.add_argument("--yes", action="store_true", help="Run without interactive confirmation")
    parser.add_argument("--no-backup", action="store_true", help="Skip creating backups")
    args = parser.parse_args()

    sqlite_path = parse_sqlite_path(settings.database_url)
    chroma_dir = Path(settings.chroma_persist_directory).resolve()
    backups_dir = (REPO_ROOT / "data" / "backups").resolve()

    print("Cleanup plan:")
    print(f"  Mode: {args.mode}")
    print(f"  SQLite DB: {sqlite_path if sqlite_path else 'Non-SQLite or unknown'}")
    print(f"  Chroma dir: {chroma_dir}")
    print(f"  Collection: {settings.chroma_collection_name}")
    if not args.yes:
        resp = input("Proceed? (y/N): ").strip().lower()
        if resp not in {"y", "yes"}:
            print("Aborted.")
            return

    # Backups
    sqlite_backup = None
    chroma_backup = None
    if not args.no_backup:
        if sqlite_path:
            sqlite_backup = backup_sqlite(sqlite_path, backups_dir)
            if sqlite_backup:
                print(f"SQLite backup: {sqlite_backup}")
        if chroma_dir.exists():
            chroma_backup = backup_chroma_dir(chroma_dir, backups_dir)
            if chroma_backup:
                print(f"Chroma backup:  {chroma_backup}")

    # Connect DB session
    session = SessionLocal()
    try:
        # Chroma cleanup
        wipe_chroma_collection()
        print("Chroma collection cleared.")

        # SQLite cleanup
        if args.mode == "vectors":
            cleanup_vectors(session)
            print("SQLite embeddings nulled (vectors mode).")
        else:
            cleanup_all(session)
            print("All test_cases deleted (all mode).")
    finally:
        session.close()

    print("Done.")
    if sqlite_backup or chroma_backup:
        print("Backups saved under:")
        if sqlite_backup:
            print(f"  - {sqlite_backup}")
        if chroma_backup:
            print(f"  - {chroma_backup}")


if __name__ == "__main__":
    main()
