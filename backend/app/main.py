import os
import sys
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Ensure project root and ml-service are on Python module search path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
ml_service_dir = project_root / "ml-service"
if str(ml_service_dir) not in sys.path:
    sys.path.insert(0, str(ml_service_dir))

# Windows PyTorch DLL path preload
try:
    torch_lib = project_root / "venv" / "Lib" / "site-packages" / "torch" / "lib"
    if torch_lib.exists() and hasattr(os, "add_dll_directory"):
        os.add_dll_directory(str(torch_lib))
except Exception:
    pass

from backend.app.database.connection import MongoDBManager
from backend.app.api import auth, verify, news, history, evidence, models_api, admin

load_dotenv()

ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:5173,http://127.0.0.1:5173,http://localhost:3000"
).split(",")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage startup and shutdown events."""
    await MongoDBManager.connect()
    yield
    await MongoDBManager.disconnect()

app = FastAPI(
    title="Smart News Authenticity Verification Portal API",
    description="Multimodal AI-powered verification service detecting fake, real, and suspicious news.",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware for React frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in ALLOWED_ORIGINS if origin.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Routers
app.include_router(auth.router)
app.include_router(verify.router)
app.include_router(news.router)
app.include_router(history.router)
app.include_router(evidence.router)
app.include_router(models_api.router)
app.include_router(admin.router)

@app.get("/api/health", tags=["Health"])
async def health_check():
    """Health check endpoint confirming API and DB status."""
    db_connected = MongoDBManager.db is not None
    return {
        "status": "healthy",
        "database_connected": db_connected,
        "service": "Smart News Authenticity Portal API",
        "version": "1.0.0"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app.main:app", host="127.0.0.1", port=8000, reload=True)
