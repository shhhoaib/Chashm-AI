"""
Chashm AI (چشم AI) - AI-Powered Eye Disease Screening Platform
Desktop Application Entry Point
"""

import sys
import os

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QFont
from app.main_window import MainWindow
from app.config import APP_NAME


def main():
    QApplication.setApplicationName(APP_NAME)
    QApplication.setOrganizationName("Chashm AI Technologies")
    QApplication.setApplicationVersion("1.0.0")

    app = QApplication(sys.argv)

    font = QFont("Segoe UI", 10)
    app.setFont(font)

    app.setStyleSheet("""
        QToolTip {
            background-color: #1B2838;
            color: #E0E6ED;
            border: 1px solid #2A3A4A;
            border-radius: 4px;
            padding: 4px 8px;
            font-size: 12px;
        }
    """)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
