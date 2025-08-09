from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import structlog
import time
from app.api.routes import api_router
from app.config.settings import settings
from app.core.database import create_tables

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()


def create_app() -> FastAPI:
    """Create and configure the FastAPI application"""
    
    app = FastAPI(
        title="Test Case Generator API",
        description="AI-powered test case generation and management system",
        version="1.0.0",
        openapi_url=f"{settings.api_prefix}/openapi.json",
        docs_url=f"{settings.api_prefix}/docs",
        redoc_url=f"{settings.api_prefix}/redoc"
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure this properly for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Request logging middleware
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        start_time = time.time()
        
        # Log request
        logger.info(
            "Request started",
            method=request.method,
            url=str(request.url),
            client_ip=request.client.host if request.client else None
        )
        
        response = await call_next(request)
        
        process_time = time.time() - start_time
        
        # Log response
        logger.info(
            "Request completed",
            method=request.method,
            url=str(request.url),
            status_code=response.status_code,
            process_time=process_time
        )
        
        return response
    
    # Exception handler
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.error(
            "Unhandled exception",
            method=request.method,
            url=str(request.url),
            error=str(exc),
            exc_info=True
        )
        
        return JSONResponse(
            status_code=500,
            content={
                "detail": "Internal server error",
                "timestamp": time.time()
            }
        )
    
    # Include API routes
    app.include_router(api_router, prefix=settings.api_prefix)
    
    return app


# Create the application instance
app = create_app()


@app.on_event("startup")
async def startup_event():
    """Application startup event"""
    logger.info("Application starting up")
    
    try:
        # Create database tables
        create_tables()
        logger.info("Database initialization completed")
    except Exception as e:
        logger.error("Failed to initialize database", error=str(e))
        raise
    
    logger.info("Application startup completed")


@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown event"""
    logger.info("Application shutting down")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
        log_level=settings.log_level.lower()
    )
