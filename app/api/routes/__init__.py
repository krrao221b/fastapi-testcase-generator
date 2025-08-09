from fastapi import APIRouter
from app.api.routes import test_cases, integrations, health

api_router = APIRouter()

# Include all route modules
api_router.include_router(health.router)
api_router.include_router(test_cases.router)
api_router.include_router(integrations.router)
