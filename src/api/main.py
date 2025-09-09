"""
FastAPI application for Healthcare Data Pipeline Phase 3.

This module provides the main FastAPI application with all necessary
middleware, routers, and configuration for the production API.
"""

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
import time
import logging
from contextlib import asynccontextmanager
from starlette_prometheus import PrometheusMiddleware, metrics

from src.config.settings import get_settings
from src.utils.logging import setup_logging

# Import routers
from src.api.routers import health_router, optimization_router, predictions_router, monitoring_router

# Setup logging
logger = setup_logging()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("🚀 Starting Healthcare Data Pipeline API...")
    settings = get_settings()
    logger.info(f"Environment: {settings.environment}")
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down Healthcare Data Pipeline API...")

def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()
    
    app = FastAPI(
        title="Healthcare Data Pipeline API",
        description="Production API for healthcare workload optimization and ML predictions",
        version="3.0.0",
        docs_url="/docs" if settings.environment != "production" else None,
        redoc_url="/redoc" if settings.environment != "production" else None,
        lifespan=lifespan
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["*"],
    )
    
    # Add trusted host middleware for production
    if settings.environment == "production":
        app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=settings.allowed_hosts
        )
    
    # Add Prometheus middleware
    app.add_middleware(PrometheusMiddleware)
    app.add_route("/metrics", metrics)
    
    # Add request logging middleware
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        start_time = time.time()
        
        # Log request
        logger.info(f"📥 {request.method} {request.url.path}")
        
        response = await call_next(request)
        
        # Log response
        process_time = time.time() - start_time
        logger.info(f"📤 {request.method} {request.url.path} - {response.status_code} - {process_time:.3f}s")
        
        # Add timing header
        response.headers["X-Process-Time"] = str(process_time)
        
        return response
    
    # Add exception handlers
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        logger.error(f"Validation error: {exc.errors()}")
        return JSONResponse(
            status_code=422,
            content={
                "detail": "Validation error",
                "errors": exc.errors()
            }
        )
    
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        logger.error(f"HTTP error {exc.status_code}: {exc.detail}")
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail}
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        logger.error(f"Unexpected error: {str(exc)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error"}
        )
    
    # Include routers
    app.include_router(health_router, prefix="/api/v1", tags=["health"])
    app.include_router(optimization_router, prefix="/api/v1/optimize", tags=["optimization"])
    app.include_router(predictions_router, prefix="/api/v1/predict", tags=["predictions"])
    app.include_router(monitoring_router, prefix="/api/v1/monitoring", tags=["monitoring"])
    
    # Root endpoint
    @app.get("/")
    async def root():
        return {
            "message": "Healthcare Data Pipeline API v3.0.0",
            "status": "operational",
            "environment": settings.environment,
            "docs": "/docs" if settings.environment != "production" else None
        }
    
    # Health check endpoint
    @app.get("/health")
    async def health_check():
        return {
            "status": "healthy",
            "timestamp": time.time(),
            "version": "3.0.0"
        }
    
    return app

# Create the application instance
app = create_app()

if __name__ == "__main__":
    import uvicorn
    settings = get_settings()
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.environment == "development",
        log_level=settings.log_level.lower()
    )
