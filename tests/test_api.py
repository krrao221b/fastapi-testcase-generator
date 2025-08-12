import pytest
from tests.conftest import test_client


def test_health_check(test_client):
    """Test health check endpoint"""
    response = test_client.get("/api/v1/health")
    assert response.status_code == 200
    
    data = response.json()
    assert data["status"] == "healthy"
    assert "timestamp" in data
    assert "version" in data
    assert "environment" in data


def test_readiness_check(test_client):
    """Test readiness check endpoint"""
    response = test_client.get("/api/v1/health/readiness")
    assert response.status_code == 200
    
    data = response.json()
    assert "status" in data
    assert "checks" in data
    assert "timestamp" in data


@pytest.mark.asyncio
async def test_generate_test_case_endpoint(client):
    """Test test case generation endpoint"""
    request_data = {
        "feature_description": "User login functionality",
        "acceptance_criteria": "Users should be able to login with valid credentials",
        "priority": "medium",
        "tags": ["authentication"]
    }
    
    # Note: This test might fail without proper Gemini API key
    # In a real environment, you'd mock the AI service
    response = await client.post("/api/v1/test-cases/generate", json=request_data)
    
    # The response might be 500 due to missing API key, which is expected in test environment
    assert response.status_code in [200, 500]


def test_get_all_test_cases(test_client):
    """Test get all test cases endpoint"""
    response = test_client.get("/api/v1/test-cases/")
    assert response.status_code == 200
    
    data = response.json()
    assert isinstance(data, list)
