from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QLabel, QProgressBar, QPushButton, QApplication
)
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QFont, QPixmap
from pathlib import Path

from app.config import ACCENT_COLOR, SECONDARY_COLOR, CARD_BG, DARK_BG, TEXT_PRIMARY, TEXT_SECONDARY


class ModelLoadingDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Chashm AI - Loading Models")
        self.setFixedSize(460, 220)
        self.setObjectName("loadingDialog")
        self.setStyleSheet(f"""
            #loadingDialog {{
                background-color: {DARK_BG};
                border: 1px solid #2A3A4A;
            }}
        """)
        self.setModal(True)
        self._setup_ui()
        self._progress = 0

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(30, 24, 30, 24)
        layout.setSpacing(12)

        logo_path = Path(__file__).resolve().parent.parent.parent.parent / "logo.png"
        logo_label = QLabel()
        if logo_path.exists():
            pixmap = QPixmap(str(logo_path))
            logo_label.setPixmap(pixmap.scaled(48, 48, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            logo_label.setText("CH")
            logo_label.setStyleSheet(f"font-size: 28px; font-weight: 800; color: {ACCENT_COLOR};")
        logo_label.setAlignment(Qt.AlignCenter)
        logo_label.setFixedHeight(56)
        layout.addWidget(logo_label)

        title = QLabel("Loading AI Models")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet(f"font-size: 20px; font-weight: 700; color: {ACCENT_COLOR};")
        layout.addWidget(title)

        self.status_label = QLabel("Initializing model engine...")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet(f"font-size: 13px; color: {TEXT_SECONDARY};")
        layout.addWidget(self.status_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setStyleSheet(f"""
            QProgressBar {{
                background-color: #2A3A4A;
                border: none;
                border-radius: 6px;
                height: 18px;
                text-align: center;
                font-size: 11px;
                color: white;
                font-weight: 600;
            }}
            QProgressBar::chunk {{
                background-color: {SECONDARY_COLOR};
                border-radius: 6px;
            }}
        """)
        layout.addWidget(self.progress_bar)

        self.model_label = QLabel("")
        self.model_label.setAlignment(Qt.AlignCenter)
        self.model_label.setStyleSheet(f"font-size: 11px; color: {TEXT_SECONDARY};")
        layout.addWidget(self.model_label)

        self.device_label = QLabel("")
        self.device_label.setAlignment(Qt.AlignCenter)
        self.device_label.setStyleSheet("font-size: 10px; color: #5A6A7A;")
        layout.addWidget(self.device_label)

        layout.addStretch()

    def update_progress(self, percent: int, status: str, model_name: str = "", device: str = ""):
        self._progress = percent
        self.progress_bar.setValue(percent)
        self.status_label.setText(status)
        if model_name:
            self.model_label.setText(model_name)
        if device:
            self.device_label.setText(f"Device: {device}")
        QApplication.processEvents()

    def set_device(self, device: str):
        self.device_label.setText(f"Device: {device}")

    def mark_complete(self):
        self.progress_bar.setValue(100)
        self.status_label.setText("All models loaded successfully!")
        self.model_label.setText("Ready for analysis")
        QApplication.processEvents()
