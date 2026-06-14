from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QLineEdit,
    QGroupBox, QFormLayout, QMessageBox, QFileDialog, QTabWidget, QCheckBox
)
from PySide6.QtCore import Qt

from app.database.connection import get_session
from app.database.models import Doctor
from app.config import APP_NAME, APP_VERSION


class SettingsView(QWidget):
    def __init__(self, doctor_id: str = None, parent=None):
        super().__init__(parent)
        self.doctor_id = doctor_id
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(16)

        title = QLabel("Settings")
        title.setStyleSheet("font-size: 24px; font-weight: 700; color: #E0E6ED;")
        layout.addWidget(title)

        tabs = QTabWidget()
        tabs.addTab(self._build_profile_tab(), "Profile")
        tabs.addTab(self._build_preferences_tab(), "Preferences")
        tabs.addTab(self._build_about_tab(), "About")
        layout.addWidget(tabs)

    def _build_profile_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(20, 20, 20, 20)

        group = QGroupBox("Doctor Profile")
        form = QFormLayout(group)
        form.setSpacing(10)

        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("Full Name")
        form.addRow("Full Name:", self.name_input)

        self.hospital_input = QLineEdit()
        self.hospital_input.setPlaceholderText("Hospital Name")
        form.addRow("Hospital:", self.hospital_input)

        self.reg_input = QLineEdit()
        self.reg_input.setPlaceholderText("Registration Number")
        self.reg_input.setReadOnly(True)
        form.addRow("Registration No:", self.reg_input)

        self.email_input = QLineEdit()
        self.email_input.setPlaceholderText("Email")
        form.addRow("Email:", self.email_input)

        self.mobile_input = QLineEdit()
        self.mobile_input.setPlaceholderText("Mobile Number")
        form.addRow("Mobile:", self.mobile_input)

        self.qual_input = QLineEdit()
        self.qual_input.setPlaceholderText("Qualification")
        form.addRow("Qualification:", self.qual_input)

        self.spec_input = QLineEdit()
        self.spec_input.setPlaceholderText("Specialization")
        form.addRow("Specialization:", self.spec_input)

        layout.addWidget(group)

        save_btn = QPushButton("Save Changes")
        save_btn.clicked.connect(self._save_profile)
        layout.addWidget(save_btn)
        layout.addStretch()

        self._load_profile()
        return tab

    def _load_profile(self):
        if not self.doctor_id:
            return
        session = get_session()
        try:
            doctor = session.query(Doctor).get(self.doctor_id)
            if doctor:
                self.name_input.setText(doctor.full_name or "")
                self.hospital_input.setText(doctor.hospital_name or "")
                self.reg_input.setText(doctor.registration_number or "")
                self.email_input.setText(doctor.email or "")
                self.mobile_input.setText(doctor.mobile_number or "")
                self.qual_input.setText(doctor.qualification or "")
                self.spec_input.setText(doctor.specialization or "")
        finally:
            session.close()

    def _save_profile(self):
        if not self.doctor_id:
            return
        session = get_session()
        try:
            doctor = session.query(Doctor).get(self.doctor_id)
            if doctor:
                doctor.full_name = self.name_input.text().strip()
                doctor.hospital_name = self.hospital_input.text().strip()
                doctor.email = self.email_input.text().strip()
                doctor.mobile_number = self.mobile_input.text().strip()
                doctor.qualification = self.qual_input.text().strip()
                doctor.specialization = self.spec_input.text().strip()
                session.commit()
                QMessageBox.information(self, "Saved", "Profile updated successfully")
        except Exception as e:
            session.rollback()
            QMessageBox.warning(self, "Error", str(e))
        finally:
            session.close()

    def _build_preferences_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(20, 20, 20, 20)

        group = QGroupBox("Application Preferences")
        form = QFormLayout(group)
        form.setSpacing(12)

        self.auto_refresh = QCheckBox("Auto-refresh dashboard")
        self.auto_refresh.setChecked(True)
        form.addRow("Dashboard:", self.auto_refresh)

        self.notifications = QCheckBox("Enable notifications")
        self.notifications.setChecked(True)
        form.addRow("Notifications:", self.notifications)

        self.offline_mode = QCheckBox("Offline AI mode (no internet required)")
        self.offline_mode.setChecked(True)
        self.offline_mode.setEnabled(False)
        form.addRow("AI Mode:", self.offline_mode)

        self.high_quality_viz = QCheckBox("High-quality visualizations")
        self.high_quality_viz.setChecked(True)
        form.addRow("Visualizations:", self.high_quality_viz)

        layout.addWidget(group)
        layout.addStretch()

        save_btn = QPushButton("Save Preferences")
        save_btn.clicked.connect(lambda: QMessageBox.information(self, "Saved", "Preferences saved"))
        layout.addWidget(save_btn)

        return tab

    def _build_about_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(20, 20, 20, 20)

        group = QGroupBox(f"About {APP_NAME}")
        glayout = QVBoxLayout(group)

        name_label = QLabel(f"<h2>{APP_NAME}</h2>")
        name_label.setTextFormat(Qt.RichText)
        name_label.setAlignment(Qt.AlignCenter)
        glayout.addWidget(name_label)

        info = QLabel(
            f"<b>Version:</b> {APP_VERSION}<br>"
            f"<b>Type:</b> AI-Powered Eye Disease Screening Platform<br>"
            f"<b>Technology:</b> PySide6, PyTorch, YOLO, Vision Transformer<br>"
            f"<b>Database:</b> PostgreSQL / SQLite<br>"
            f"<br>"
            f"<b>Features:</b><br>"
            f"&bull; AI-Powered Disease Detection<br>"
            f"&bull; Explainable AI (Grad-CAM, SHAP)<br>"
            f"&bull; Ensemble Model Architecture<br>"
            f"&bull; Automated PDF Reports<br>"
            f"&bull; QR Code Verification<br>"
            f"&bull; Multi-Hospital Support<br>"
            f"&bull; Offline AI Mode<br>"
            f"<br>"
            f"<i>AI assists doctors - it does not replace them.<br>"
            f"Always consult a licensed ophthalmologist.</i>"
        )
        info.setTextFormat(Qt.RichText)
        info.setWordWrap(True)
        glayout.addWidget(info)

        copyright_label = QLabel(f"© 2025 Chashm AI Technologies. All rights reserved.")
        copyright_label.setAlignment(Qt.AlignCenter)
        copyright_label.setStyleSheet("color: #5A6A7A; font-size: 10px;")
        glayout.addWidget(copyright_label)

        layout.addWidget(group)
        layout.addStretch()
        return tab
