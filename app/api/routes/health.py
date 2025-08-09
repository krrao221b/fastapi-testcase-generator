from fastapi import APIRouter, status
from pydantic import BaseModel
from datetime import datetime
from app.config.settings import settings

router = APIRouter(prefix="/health", tags=["health"])


class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
    version: str
    environment: str


@router.get("/", response_model=HealthResponse, status_code=status.HTTP_200_OK)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow(),
        version="1.0.0",
        environment=settings.environment
    )


@router.get("/readiness")
async def readiness_check():
    """Readiness check endpoint"""
    # Here you can add checks for database connectivity, external services, etc.
    checks = {
        "database": "ok",
        "chromadb": "ok",
        "openai": "ok" if settings.openai_api_key else "not_configured"
    }
    
    all_ok = all(status == "ok" for status in checks.values())
    
    return {
        "status": "ready" if all_ok else "not_ready",
        "checks": checks,
        "timestamp": datetime.utcnow()
    }
