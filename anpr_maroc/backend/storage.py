"""Stockage des matricules validés.

Backend cible : MySQL (table ``plates``). Un repli SQLite est fourni pour que
la chaîne complète (capture → validation → stockage → API) reste exécutable
tant que les identifiants MySQL ne sont pas fournis. Le schéma et l'API Python
sont strictement identiques dans les deux cas, donc basculer sur MySQL ne
demande que de renseigner les variables d'environnement ci-dessous.

Configuration (variables d'environnement, cf. .env.example) :
    ANPR_DB_BACKEND   mysql | sqlite | auto  (défaut: auto -> mysql si joignable)
    ANPR_DB_HOST      défaut 127.0.0.1
    ANPR_DB_PORT      défaut 3306
    ANPR_DB_USER      défaut anpr
    ANPR_DB_PASSWORD  (obligatoire pour MySQL)
    ANPR_DB_NAME      défaut anpr
    ANPR_DB_SQLITE    chemin du fichier de repli (défaut: plates.db)
"""

from __future__ import annotations

import os
import sqlite3
import typing as t
from datetime import datetime
from pathlib import Path

try:  # pragma: no cover - dépendance optionnelle
    import pymysql
except ImportError:  # pragma: no cover
    pymysql = None

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover
    pass


SQLITE_SCHEMA = """
CREATE TABLE IF NOT EXISTS plates (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    matricule          TEXT      NOT NULL,
    confiance          REAL      NOT NULL DEFAULT 0,
    timestamp          TIMESTAMP NOT NULL,
    chemin_image_crop  TEXT
)
"""

MYSQL_SCHEMA = """
CREATE TABLE IF NOT EXISTS plates (
    id                 INT AUTO_INCREMENT PRIMARY KEY,
    matricule          VARCHAR(32)  NOT NULL,
    confiance          FLOAT        NOT NULL DEFAULT 0,
    timestamp          DATETIME     NOT NULL,
    chemin_image_crop  VARCHAR(512) NULL,
    INDEX idx_plates_timestamp (timestamp),
    INDEX idx_plates_matricule (matricule)
) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci
"""


class PlateStore:
    """Accès en écriture/lecture à la table ``plates``.

    L'instance ouvre sa connexion à la construction et la garde ouverte ;
    ``insert_plate`` est appelé depuis la boucle live, ``recent`` depuis l'API.
    """

    def __init__(
        self,
        backend: t.Optional[str] = None,
        sqlite_path: t.Optional[str] = None,
    ) -> None:
        self.backend = (backend or os.getenv("ANPR_DB_BACKEND", "auto")).lower()
        self.sqlite_path = sqlite_path or os.getenv("ANPR_DB_SQLITE", "plates.db")
        self.conn: t.Any = None
        self.active_backend = ""
        self.last_error = ""
        self._connect()

    # ------------------------------------------------------------------ setup

    def _mysql_params(self) -> dict:
        return {
            "host": os.getenv("ANPR_DB_HOST", "127.0.0.1"),
            "port": int(os.getenv("ANPR_DB_PORT", "3306")),
            "user": os.getenv("ANPR_DB_USER", "anpr"),
            "password": os.getenv("ANPR_DB_PASSWORD", ""),
            "database": os.getenv("ANPR_DB_NAME", "anpr"),
            "charset": "utf8mb4",
            "autocommit": True,
        }

    def _connect_mysql(self) -> None:
        if pymysql is None:
            raise RuntimeError("PyMySQL n'est pas installé")
        self.conn = pymysql.connect(**self._mysql_params())
        with self.conn.cursor() as cur:
            cur.execute(MYSQL_SCHEMA)
        self.active_backend = "mysql"

    def _connect_sqlite(self) -> None:
        path = Path(self.sqlite_path)
        if path.parent != Path(""):
            path.parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False : la boucle live et l'API peuvent écrire/lire
        # depuis des threads différents.
        self.conn = sqlite3.connect(str(path), check_same_thread=False)
        self.conn.execute(SQLITE_SCHEMA)
        self.conn.commit()
        self.active_backend = "sqlite"

    def _connect(self) -> None:
        if self.backend == "sqlite":
            self._connect_sqlite()
            return
        try:
            self._connect_mysql()
        except Exception as exc:
            self.last_error = str(exc)
            if self.backend == "mysql":
                raise
            # backend == 'auto' : repli silencieux, l'appelant affiche
            # `describe()` pour signaler quel backend est réellement actif.
            self._connect_sqlite()

    # ------------------------------------------------------------------- API

    def describe(self) -> str:
        if self.active_backend == "mysql":
            p = self._mysql_params()
            return f"MySQL {p['user']}@{p['host']}:{p['port']}/{p['database']}"
        suffix = f" (repli, MySQL indisponible : {self.last_error})" if self.last_error else ""
        return f"SQLite {self.sqlite_path}{suffix}"

    def _placeholder(self) -> str:
        return "%s" if self.active_backend == "mysql" else "?"

    def insert_plate(
        self,
        matricule: str,
        confiance: float,
        chemin_image_crop: t.Optional[str] = None,
        timestamp: t.Optional[datetime] = None,
    ) -> int:
        """Insère un matricule validé et retourne son id."""
        ts = timestamp or datetime.now()
        ph = self._placeholder()
        sql = (
            "INSERT INTO plates (matricule, confiance, timestamp, chemin_image_crop) "
            f"VALUES ({ph}, {ph}, {ph}, {ph})"
        )
        values = (matricule, float(confiance), ts, chemin_image_crop)
        if self.active_backend == "mysql":
            with self.conn.cursor() as cur:
                cur.execute(sql, values)
                return int(cur.lastrowid)
        cur = self.conn.execute(sql, (matricule, float(confiance), ts.isoformat(sep=" ", timespec="seconds"), chemin_image_crop))
        self.conn.commit()
        return int(cur.lastrowid)

    def recent(self, limit: int = 20) -> list[dict]:
        """Retourne les ``limit`` dernières détections, plus récente d'abord."""
        limit = max(1, min(int(limit), 1000))
        sql = (
            "SELECT id, matricule, confiance, timestamp, chemin_image_crop "
            f"FROM plates ORDER BY id DESC LIMIT {limit}"
        )
        if self.active_backend == "mysql":
            with self.conn.cursor() as cur:
                cur.execute(sql)
                rows = cur.fetchall()
        else:
            rows = self.conn.execute(sql).fetchall()

        out = []
        for row in rows:
            ts = row[3]
            out.append(
                {
                    "id": int(row[0]),
                    "matricule": row[1],
                    "confiance": float(row[2]),
                    "timestamp": ts.isoformat() if isinstance(ts, datetime) else str(ts),
                    "chemin_image_crop": row[4],
                }
            )
        return out

    def close(self) -> None:
        try:
            if self.conn is not None:
                self.conn.close()
        except Exception:
            pass


_store: t.Optional[PlateStore] = None


def get_store() -> PlateStore:
    """Instance partagée (boucle live et API dans le même process)."""
    global _store
    if _store is None:
        _store = PlateStore()
    return _store
