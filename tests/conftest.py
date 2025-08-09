import pytest
import asyncio
from httpx import AsyncClient
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
async def client():
    """Create test client"""
    # Create tables
    Base.metadata.create_all(bind=engine)
    
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac
    
    # Clean up
    Base.metadata.drop_all(bind=engine)

@pytest.fixture
def test_client():
    """Synchronous test client for simple tests"""
    return TestClient(app)
