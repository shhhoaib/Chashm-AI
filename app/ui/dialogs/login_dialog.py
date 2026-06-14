from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QMessageBox, QWidget, QSpacerItem, QSizePolicy, QFrame,
    QApplication
)
from PySide6.QtCore import Qt, Signal, QRect
from PySide6.QtGui import QFont, QPixmap, QScreen
from pathlib import Path

from app.database.connection import get_session
from app.services.auth_service import AuthService
from app.config import PRIMARY_COLOR, SECONDARY_COLOR, ACCENT_COLOR, DARK_BG, CARD_BG, APP_NAME


class LoginDialog(QDialog):
    login_success = Signal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"{APP_NAME} - Login")
        self.setFixedSize(420, 520)
        self.setObjectName("loginDialog")
        self.setStyleSheet(f"""
            #loginDialog {{
                background-color: {DARK_BG};
            }}
        """)
        self._setup_ui()
        self._center_on_screen()

    def _center_on_screen(self):
        screen = QApplication.primaryScreen()
        if screen:
            geo = screen.availableGeometry()
            x = geo.x() + (geo.width() - self.width()) // 2
            y = geo.y() + (geo.height() - self.height()) // 2
            self.move(x, y)
        self.raise_()
        self.activateWindow()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(40, 40, 40, 40)
        layout.setSpacing(0)

        layout.addStretch(1)

        logo_path = Path(__file__).resolve().parent.parent.parent.parent / "logo.png"
        logo_label = QLabel()
        if logo_path.exists():
            pixmap = QPixmap(str(logo_path))
            logo_label.setPixmap(pixmap.scaled(56, 56, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            logo_label.setText(APP_NAME[:2])
            logo_label.setStyleSheet(f"font-size: 32px; font-weight: 800; color: {ACCENT_COLOR};")
        logo_label.setAlignment(Qt.AlignCenter)
        logo_label.setFixedHeight(64)
        layout.addWidget(logo_label)

        title = QLabel(APP_NAME)
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet(f"font-size: 32px; font-weight: 700; color: {ACCENT_COLOR};")
        layout.addWidget(title)

        subtitle = QLabel("Eye Disease Screening Platform")
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setStyleSheet("font-size: 12px; color: #8892A0; margin-bottom: 30px;")
        layout.addWidget(subtitle)

        layout.addSpacing(10)

        form_card = QFrame()
        form_card.setObjectName("formCard")
        form_card.setStyleSheet(f"""
            #formCard {{
                background-color: {CARD_BG};
                border-radius: 12px;
                padding: 24px;
                border: 1px solid #2A3A4A;
            }}
        """)
        form_layout = QVBoxLayout(form_card)
        form_layout.setContentsMargins(20, 20, 20, 20)
        form_layout.setSpacing(12)

        email_label = QLabel("Email")
        email_label.setStyleSheet("font-size: 12px; color: #8892A0; font-weight: 600;")
        form_layout.addWidget(email_label)

        self.email_input = QLineEdit()
        self.email_input.setPlaceholderText("doctor@hospital.com")
        self.email_input.setText("admin@chashm.ai")
        form_layout.addWidget(self.email_input)

        pwd_label = QLabel("Password")
        pwd_label.setStyleSheet("font-size: 12px; color: #8892A0; font-weight: 600;")
        form_layout.addWidget(pwd_label)

        self.password_input = QLineEdit()
        self.password_input.setPlaceholderText("Enter password")
        self.password_input.setText("admin123")
        self.password_input.setEchoMode(QLineEdit.Password)
        self.password_input.returnPressed.connect(self._login)
        form_layout.addWidget(self.password_input)

        form_layout.addSpacing(10)

        self.login_btn = QPushButton("Sign In")
        self.login_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {SECONDARY_COLOR};
                color: white;
                border: none;
                border-radius: 8px;
                padding: 12px;
                font-size: 14px;
                font-weight: 600;
            }}
            QPushButton:hover {{
                background-color: {ACCENT_COLOR};
            }}
        """)
        self.login_btn.clicked.connect(self._login)
        form_layout.addWidget(self.login_btn)

        self.error_label = QLabel("")
        self.error_label.setAlignment(Qt.AlignCenter)
        self.error_label.setStyleSheet("color: #FF4757; font-size: 11px;")
        self.error_label.setFixedHeight(20)
        form_layout.addWidget(self.error_label)

        layout.addWidget(form_card)

        layout.addStretch(1)

        footer = QLabel("v1.0.0 | AI-Assisted Eye Screening")
        footer.setAlignment(Qt.AlignCenter)
        footer.setStyleSheet("font-size: 10px; color: #5A6A7A;")
        layout.addWidget(footer)

    def _login(self):
        email = self.email_input.text().strip()
        password = self.password_input.text().strip()

        if not email or not password:
            self.error_label.setText("Please enter email and password")
            return

        self.login_btn.setEnabled(False)
        self.login_btn.setText("Signing in...")

        session = get_session()
        try:
            auth_service = AuthService(session)
            token, error = auth_service.authenticate(email, password)
            if error:
                self.error_label.setText(error)
                self.login_btn.setEnabled(True)
                self.login_btn.setText("Sign In")
                return

            doctor, _ = auth_service.verify_token(token)
            if doctor:
                self.login_success.emit({
                    "doctor_id": doctor.id,
                    "full_name": doctor.full_name,
                    "hospital_name": doctor.hospital_name,
                    "email": doctor.email,
                    "registration_number": doctor.registration_number,
                    "token": token,
                })
                self.accept()
            else:
                self.error_label.setText("Authentication failed")
        except Exception as e:
            self.error_label.setText(f"Error: {str(e)}")
        finally:
            session.close()
            self.login_btn.setEnabled(True)
            self.login_btn.setText("Sign In")
