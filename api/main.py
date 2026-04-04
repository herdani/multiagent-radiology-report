import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import make_asgi_app

from api.models.database import init_db
from api.routes.health import router as health_router
from api.routes.reports import router as reports_router
from api.routes.pipeline import router as pipeline_router
from api.routes.compliance import router as compliance_router

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        logger.info("Starting up — initializing database...")
        init_db()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.warning("Database unavailable at startup: %s — will retry on first request", e)
    yield


app = FastAPI(
    title="Radiology AI",
    description="Production-grade multi-agent AI radiology report generation",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# medical middleware — one line does GDPR + rate limiting + security headers
try:
    from medical_middleware import setup_middleware
    from medical_middleware.config import MiddlewareConfig

    setup_middleware(
        app,
        MiddlewareConfig(
            data_retention_seconds=7776000,  # 90 days
            require_consent_header=False,
            rate_limit_predict="10/minute",
            rate_limit_default="30/minute",
            app_name="radiology-ai",
        ),
    )
    logger.info("Medical middleware loaded — GDPR + rate limiting active")
except ImportError:
    logger.warning("medical-ai-middleware not installed — skipping")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:7860"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

app.include_router(health_router)
app.include_router(reports_router)
app.include_router(pipeline_router)
app.include_router(compliance_router)
