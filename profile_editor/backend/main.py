import os
import json
from pathlib import Path
from typing import Any

import yaml
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

# Paths
BASE_DIR = Path(__file__).parent
PROFILES_DIR = BASE_DIR / "../../profiles"
FRONTEND_DIST = BASE_DIR / "../frontend/dist"

app = FastAPI(title="Profile Editor API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class NewProfile(BaseModel):
    name: str
    description: str


def profile_path(name: str) -> Path:
    # Sanitize name to prevent path traversal
    safe_name = Path(name).name
    return PROFILES_DIR / f"{safe_name}.yaml"


def load_yaml(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def save_yaml(path: Path, data: dict) -> None:
    with open(path, "w") as f:
        yaml.dump(data, f, allow_unicode=True, sort_keys=False, default_flow_style=False)


# ── API routes ────────────────────────────────────────────────────────────────

@app.get("/api/profiles")
def list_profiles():
    """Return sorted list of profile names (without .yaml extension)."""
    if not PROFILES_DIR.exists():
        return []
    return sorted(
        p.stem for p in PROFILES_DIR.glob("*.yaml")
    )


@app.get("/api/profiles/{name}")
def get_profile(name: str):
    path = profile_path(name)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Profile '{name}' not found")
    return load_yaml(path)


@app.put("/api/profiles/{name}")
async def update_profile(name: str, request: Request):
    """Write profile data back to YAML. Accepts any JSON."""
    path = profile_path(name)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Profile '{name}' not found")
    data = await request.json()
    save_yaml(path, data)
    return {"status": "ok"}


@app.post("/api/profiles")
def create_profile(body: NewProfile):
    path = profile_path(body.name)
    if path.exists():
        raise HTTPException(status_code=409, detail=f"Profile '{body.name}' already exists")
    initial = {"name": body.name, "description": body.description}
    save_yaml(path, initial)
    return {"status": "created", "name": body.name}


@app.delete("/api/profiles/{name}")
def delete_profile(name: str):
    path = profile_path(name)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Profile '{name}' not found")
    path.unlink()
    return {"status": "deleted"}


# ── Static frontend (production) ──────────────────────────────────────────────

if FRONTEND_DIST.exists():
    app.mount("/assets", StaticFiles(directory=str(FRONTEND_DIST / "assets")), name="assets")

    @app.get("/", include_in_schema=False)
    @app.get("/{path:path}", include_in_schema=False)
    def serve_frontend(path: str = ""):
        index = FRONTEND_DIST / "index.html"
        return FileResponse(str(index))
