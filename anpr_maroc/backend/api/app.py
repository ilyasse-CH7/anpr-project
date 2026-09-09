"""API de consultation des matricules détectés (FastAPI).

Endpoints :
    GET /plates?limit=N   -> N dernières détections
    GET /health           -> état du service et backend de stockage actif
    GET /images/<fichier>  -> crop de plaque (montage statique)

Pas d'authentification à ce stade (à ajouter avant toute exposition hors LAN).

Lancement :
    source .venv/bin/activate && export PYTHONPATH=$PWD
    python -m anpr_maroc.scripts.run_server
    # ou : uvicorn anpr_maroc.backend.api.app:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, Query, Request
from fastapi.staticfiles import StaticFiles

from anpr_maroc.backend.storage import get_store

# Dossier où live_camera.py écrit les crops des plaques validées.
CROPS_DIR = Path(os.getenv("ANPR_CROPS_DIR", "data/pipeline_output/live"))
CROPS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="ANPR Maroc — API de consultation",
    version="1.0.0",
    description="Consultation des matricules marocains détectés sur le flux RTSP.",
)
app.mount("/images", StaticFiles(directory=str(CROPS_DIR)), name="images")


def _image_url(request: Request, chemin: str | None) -> str | None:
    """Transforme un chemin disque en URL servie par /images."""
    if not chemin:
        return None
    name = Path(chemin).name
    if not (CROPS_DIR / name).is_file():
        return None
    return str(request.base_url).rstrip("/") + f"/images/{name}"


@app.get("/health")
def health() -> dict:
    store = get_store()
    return {"status": "ok", "stockage": store.describe()}


@app.get("/plates")
def list_plates(request: Request, limit: int = Query(20, ge=1, le=1000)) -> dict:
    """Retourne les `limit` dernières détections, la plus récente d'abord."""
    store = get_store()
    rows = store.recent(limit)
    return {
        "count": len(rows),
        "plates": [
            {
                "id": row["id"],
                "matricule": row["matricule"],
                "confiance": round(row["confiance"], 4),
                "timestamp": row["timestamp"],
                "url_image": _image_url(request, row["chemin_image_crop"]),
            }
            for row in rows
        ],
    }
