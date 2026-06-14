import os
from pathlib import Path
from dotenv import load_dotenv

try:
    import requests
    _HAS_REQUESTS = bool(requests)
except ImportError:
    _HAS_REQUESTS = False

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
ASSETS_DIR = BASE_DIR / "assets"
ICONS_DIR = ASSETS_DIR / "icons"
STYLES_DIR = ASSETS_DIR / "styles"
FONTS_DIR = ASSETS_DIR / "fonts"

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///chashm_ai.db")
SECRET_KEY = os.getenv("SECRET_KEY", "chashm-ai-secret-key-change-in-production")
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = 24

PRIMARY_COLOR = "#001F54"
SECONDARY_COLOR = "#0A84FF"
ACCENT_COLOR = "#33C3FF"
WHITE = "#FFFFFF"
DARK_BG = "#0D1B2A"
CARD_BG = "#1B2838"
TEXT_PRIMARY = "#E0E6ED"
TEXT_SECONDARY = "#8892A0"
SUCCESS_COLOR = "#2ED573"
WARNING_COLOR = "#FFA502"
DANGER_COLOR = "#FF4757"

TRAINED_DIR = BASE_DIR / "data" / "trained_models"
REPORTS_DIR = BASE_DIR / "data" / "reports"
QRCODES_DIR = BASE_DIR / "data" / "qrcodes"
SCANS_DIR = BASE_DIR / "data" / "scans"
HEATMAPS_DIR = BASE_DIR / "data" / "heatmaps"
ANNOTATED_DIR = BASE_DIR / "data" / "annotated"
SEGMENTATIONS_DIR = BASE_DIR / "data" / "segmentations"
GRADCAMS_DIR = BASE_DIR / "data" / "gradcams"
VIZ_DIR = BASE_DIR / "data" / "viz"

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "qwen/qwen2.5-vl-72b-instruct:free")
OPENROUTER_ENABLED = bool(OPENROUTER_API_KEY) and _HAS_REQUESTS

APP_NAME = "Chashm AI"
APP_VERSION = "1.0.0"
COMPANY_NAME = "Chashm AI Technologies"
