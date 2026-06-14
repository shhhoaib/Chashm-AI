from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QLineEdit,
    QTableWidget, QTableWidgetItem, QHeaderView, QDialog, QFormLayout, QMessageBox
)
from PySide6.QtCore import Qt
from app.database.connection import get_session
from app.database.models import Doctor
from app.services.auth_service import AuthService


class DoctorView(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(16)

        header_layout = QHBoxLayout()
        title = QLabel("Doctors")
        title.setStyleSheet("font-size: 24px; font-weight: 700; color: #E0E6ED;")
        header_layout.addWidget(title)
        header_layout.addStretch()
        add_btn = QPushButton("+ Register Doctor")
        add_btn.clicked.connect(self._add_doctor)
        header_layout.addWidget(add_btn)
        layout.addLayout(header_layout)

        self.table = QTableWidget()
        self.table.setColumnCount(7)
        self.table.setHorizontalHeaderLabels([
            "Name", "Hospital", "Reg No", "Email", "Mobile", "Specialization", "Status"
        ])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setVisible(False)
        layout.addWidget(self.table)

        self._load_doctors()

    def _load_doctors(self):
        session = get_session()
        try:
            doctors = session.query(Doctor).order_by(Doctor.created_at.desc()).all()
            self.table.setRowCount(len(doctors))
            for i, d in enumerate(doctors):
                self.table.setItem(i, 0, QTableWidgetItem(d.full_name))
                self.table.setItem(i, 1, QTableWidgetItem(d.hospital_name))
                self.table.setItem(i, 2, QTableWidgetItem(d.registration_number))
                self.table.setItem(i, 3, QTableWidgetItem(d.email))
                self.table.setItem(i, 4, QTableWidgetItem(d.mobile_number))
                self.table.setItem(i, 5, QTableWidgetItem(d.specialization or ""))
                self.table.setItem(i, 6, QTableWidgetItem("Active" if d.is_active else "Inactive"))
        finally:
            session.close()

    def _add_doctor(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Register Doctor")
        dialog.setFixedSize(450, 500)
        dialog.setModal(True)
        dialog.setStyleSheet("QDialog { background-color: #0D1B2A; } QLabel { color: #E0E6ED; }")
        layout = QVBoxLayout(dialog)
        form = QFormLayout()
        form.setSpacing(10)

        name_input = QLineEdit(); name_input.setPlaceholderText("Full Name")
        hospital_input = QLineEdit(); hospital_input.setPlaceholderText("Hospital Name")
        reg_input = QLineEdit(); reg_input.setPlaceholderText("Registration Number")
        mobile_input = QLineEdit(); mobile_input.setPlaceholderText("Mobile Number")
        email_input = QLineEdit(); email_input.setPlaceholderText("Email Address")
        qual_input = QLineEdit(); qual_input.setPlaceholderText("Qualification")
        spec_input = QLineEdit(); spec_input.setPlaceholderText("Specialization")
        pwd_input = QLineEdit(); pwd_input.setPlaceholderText("Password"); pwd_input.setEchoMode(QLineEdit.Password)

        form.addRow("Full Name:", name_input)
        form.addRow("Hospital:", hospital_input)
        form.addRow("Reg No:", reg_input)
        form.addRow("Mobile:", mobile_input)
        form.addRow("Email:", email_input)
        form.addRow("Qualification:", qual_input)
        form.addRow("Specialization:", spec_input)
        form.addRow("Password:", pwd_input)
        layout.addLayout(form)

        btn_layout = QHBoxLayout()
        cancel_btn = QPushButton("Cancel")
        cancel_btn.setStyleSheet("background-color: transparent; border: 1px solid #2A3A4A; color: #8892A0;")
        cancel_btn.clicked.connect(dialog.reject)
        save_btn = QPushButton("Register")
        save_btn.setStyleSheet("background-color: #0A84FF; color: white;")
        btn_layout.addWidget(cancel_btn)
        btn_layout.addWidget(save_btn)
        layout.addLayout(btn_layout)

        def save():
            data = {
                "full_name": name_input.text().strip(),
                "hospital_name": hospital_input.text().strip(),
                "registration_number": reg_input.text().strip(),
                "mobile_number": mobile_input.text().strip(),
                "email": email_input.text().strip(),
                "qualification": qual_input.text().strip(),
                "specialization": spec_input.text().strip(),
                "password": pwd_input.text().strip(),
            }
            session = get_session()
            try:
                auth = AuthService(session)
                doctor, error = auth.register_doctor(data)
                if error:
                    QMessageBox.warning(dialog, "Error", error)
                else:
                    QMessageBox.information(dialog, "Success", f"Doctor registered: {doctor.full_name}")
                    self._load_doctors()
                    dialog.accept()
            finally:
                session.close()

        save_btn.clicked.connect(save)
        dialog.exec()

    def refresh(self):
        self._load_doctors()
