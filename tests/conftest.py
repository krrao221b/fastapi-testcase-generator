import pytest
import asyncio
from httpx import AsyncClient, ASGITransport
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from main import app
from app.core.database import get_database
from app.models.database import Base

# Test database
SQLALCHEMY_DATABASE_URL = "sqlite:///./test.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def override_get_db():
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()

app.dependency_overrides[get_database] = override_get_db

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(scope="session")
def client(event_loop):
    """Create test client using the session event loop so async tests can await it."""
    # Create tables
    Base.metadata.create_all(bind=engine)

    transport = ASGITransport(app=app)
    ac = AsyncClient(transport=transport, base_url="http://test")
    # Enter AsyncClient context on the provided event loop
    event_loop.run_until_complete(ac.__aenter__())

    yield ac

    # Teardown
    event_loop.run_until_complete(ac.__aexit__(None, None, None))
    Base.metadata.drop_all(bind=engine)

@pytest.fixture
def test_client():
    """Synchronous test client for simple tests"""
    return TestClient(app)
