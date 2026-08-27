"""Central configuration for anpr_maroc
Loads from environment; optional python-dotenv support for local .env files.
"""
import os
try:
    from dotenv import load_dotenv
    load_dotenv()  # load .env if present
except Exception:
    pass

CAMERA_RTSP_URL = os.getenv('CAMERA_RTSP_URL')
CAMERA_IP = os.getenv('CAMERA_IP')
RTSP_USER = os.getenv('RTSP_USER')
RTSP_PASS = os.getenv('RTSP_PASS')
SERVER_URL = os.getenv('SERVER_URL')
JWT_SECRET = os.getenv('JWT_SECRET')
DATABASE_URL = os.getenv('DATABASE_URL')
