import os
import re
from pathlib import Path


def validate_email(email: str) -> bool:
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))


def validate_phone(phone: str) -> bool:
    pattern = r'^\+?1?\d{10,15}$'
    return bool(re.match(pattern, re.sub(r'[\s\-\(\)]', '', phone)))


def validate_registration_number(reg: str) -> bool:
    return bool(re.match(r'^[A-Za-z0-9\-]+$', reg))


def ensure_dir(path: str) -> str:
    Path(path).mkdir(parents=True, exist_ok=True)
    return path


def get_file_size_display(size_bytes: int) -> str:
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f} MB"


ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".dcm", ".bmp", ".tiff", ".tif"}
ALLOWED_IMAGE_MIMETYPES = {"image/jpeg", "image/png", "image/bmp", "image/tiff"}


def is_allowed_image(filename: str) -> bool:
    ext = Path(filename).suffix.lower()
    return ext in ALLOWED_IMAGE_EXTENSIONS


def get_disease_color(severity_grade: int) -> str:
    if severity_grade >= 4:
        return "#FF4757"
    elif severity_grade >= 3:
        return "#FF6B81"
    elif severity_grade >= 2:
        return "#FFA502"
    elif severity_grade >= 1:
        return "#2ED573"
    return "#0A84FF"


def get_risk_color(risk_level: str) -> str:
    colors = {"High": "#FF4757", "Medium": "#FFA502", "Low": "#2ED573"}
    return colors.get(risk_level, "#8892A0")
