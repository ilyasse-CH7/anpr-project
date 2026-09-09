"""Lance l'API de consultation ANPR.

    source .venv/bin/activate && export PYTHONPATH=$PWD
    python -m anpr_maroc.scripts.run_server --port 8000
"""

from __future__ import annotations

import argparse
import os


def main() -> None:
    parser = argparse.ArgumentParser(description="Serveur API ANPR (FastAPI/uvicorn)")
    parser.add_argument("--host", default=os.getenv("ANPR_API_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.getenv("ANPR_API_PORT", "8000")))
    parser.add_argument("--reload", action="store_true", help="Rechargement à chaud (développement)")
    args = parser.parse_args()

    import uvicorn

    uvicorn.run(
        "anpr_maroc.backend.api.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
