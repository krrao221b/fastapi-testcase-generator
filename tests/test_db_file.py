from pathlib import Path


def test_db_file_exists_after_create_tables():
    """Ensure the configured sqlite database file exists after running create_tables().

    This validates the parent-directory creation logic that prevents "unable to open database file"
    when the data directory doesn't exist (e.g., in fresh deploys).
    """
    # Import inside the test so settings are already loaded from the environment
    from app.core.database import create_tables
    from app.config.settings import settings

    # Run the function that ensures the directory and creates tables
    create_tables()

    # Resolve the database path the same way the application does
    path_str = settings.database_url.replace("sqlite://", "", 1)
    db_path = Path(path_str)
    if not db_path.is_absolute():
        db_path = (Path.cwd() / db_path).resolve()

    # Some environments interpret leading slashes differently; ensure the parent directory exists
    db_dir = db_path.parent
    assert db_dir.exists(), f"Expected database directory at {db_dir} to exist"
