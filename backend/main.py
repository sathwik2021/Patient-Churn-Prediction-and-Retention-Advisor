"""
Patient Churn Prediction — FastAPI Server
==========================================
Run: uvicorn main:app --reload --port 8000
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response
import os

from models.predictor import predictor
from routes.predict import router as predict_router
from routes.auth import router as auth_router
import database  # noqa: F401 — triggers init_db() on import


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model artifacts on startup."""
    predictor.load()
    print("[OK] Patient Churn Prediction Model loaded successfully")
    yield
    print("[STOP] Shutting down")


app = FastAPI(
    title="Patient Churn Prediction and Retention Advisor",
    description="AI-powered churn probability %, churn reason diagnosis, and retention advice engine",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict_router)
app.include_router(auth_router)

frontend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend"))
static_dir = os.path.join(os.path.dirname(__file__), "static")

# Mount frontend directory for direct file access
if os.path.exists(frontend_dir):
    app.mount("/frontend", StaticFiles(directory=frontend_dir), name="frontend")
elif os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/favicon.ico")
async def favicon():
    favicon_file = os.path.join(frontend_dir, "icon.svg")
    if os.path.exists(favicon_file):
        return Response(content=open(favicon_file, "rb").read(), media_type="image/svg+xml")
    return Response(status_code=204)


@app.get("/")
async def serve_index():
    index_file = os.path.join(frontend_dir, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    fallback_static = os.path.join(static_dir, "index.html")
    if os.path.exists(fallback_static):
        return FileResponse(fallback_static)
    return {"message": "Patient Churn Prediction API is running"}
