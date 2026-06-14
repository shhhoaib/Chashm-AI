from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QLineEdit,
    QTableWidget, QTableWidgetItem, QHeaderView, QDialog, QFormLayout,
    QComboBox, QTextEdit, QMessageBox, QGroupBox, QFrame, QSpacerItem,
    QSizePolicy, QTabWidget
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont

from app.database.connection import get_session
from app.database.models import Patient
from app.services.patient_service import PatientService


class PatientFormDialog(QDialog):
    def __init__(self, doctor_id: str, patient_data: dict = None, parent=None):
        super().__init__(parent)
        self.doctor_id = doctor_id
        self.patient_data = patient_data
        self.result_data = None
        self.setWindowTitle("Add Patient" if not patient_data else "Edit Patient")
        self.setFixedSize(500, 550)
        self.setModal(True)
        self.setStyleSheet("""
            QDialog { background-color: #0D1B2A; }
            QLabel { color: #E0E6ED; font-size: 12px; }
        """)
        self._setup_ui()
        if patient_data:
            self._populate_form()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 20)

        title = QLabel("Patient Information")
        title.setStyleSheet("font-size: 18px; font-weight: 700; color: #33C3FF;")
        layout.addWidget(title)

        form = QFormLayout()
        form.setSpacing(10)
        form.setLabelAlignment(Qt.AlignRight)

        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("Full Name")
        form.addRow("Name:", self.name_input)

        age_layout = QHBoxLayout()
        self.age_input = QLineEdit()
        self.age_input.setPlaceholderText("Age")
        self.age_input.setFixedWidth(100)
        self.gender_combo = QComboBox()
        self.gender_combo.addItems(["", "Male", "Female", "Other"])
        self.gender_combo.setFixedWidth(120)
        age_layout.addWidget(self.age_input)
        age_layout.addWidget(self.gender_combo)
        age_layout.addStretch()
        form.addRow("Age / Gender:", age_layout)

        self.mobile_input = QLineEdit()
        self.mobile_input.setPlaceholderText("+1234567890")
        form.addRow("Mobile:", self.mobile_input)

        self.address_input = QTextEdit()
        self.address_input.setPlaceholderText("Address")
        self.address_input.setFixedHeight(60)
        form.addRow("Address:", self.address_input)

        self.history_input = QTextEdit()
        self.history_input.setPlaceholderText("Past medical history")
        self.history_input.setFixedHeight(60)
        form.addRow("Medical History:", self.history_input)

        self.diabetes_combo = QComboBox()
        self.diabetes_combo.addItems(["", "No", "Type 1", "Type 2", "Pre-diabetic", "Unknown"])
        form.addRow("Diabetes:", self.diabetes_combo)

        self.bp_combo = QComboBox()
        self.bp_combo.addItems(["", "Normal", "Pre-hypertension", "Stage 1", "Stage 2", "Unknown"])
        form.addRow("Blood Pressure:", self.bp_combo)

        layout.addLayout(form)
        layout.addStretch()

        btn_layout = QHBoxLayout()
        cancel_btn = QPushButton("Cancel")
        cancel_btn.setStyleSheet("background-color: transparent; border: 1px solid #2A3A4A; color: #8892A0;")
        cancel_btn.clicked.connect(self.reject)
        save_btn = QPushButton("Save Patient")
        save_btn.setStyleSheet("background-color: #0A84FF; color: white; font-weight: 600;")
        save_btn.clicked.connect(self._save)
        btn_layout.addWidget(cancel_btn)
        btn_layout.addWidget(save_btn)
        layout.addLayout(btn_layout)

    def _populate_form(self):
        self.name_input.setText(self.patient_data.get("name", ""))
        self.age_input.setText(str(self.patient_data.get("age", "")))
        idx = self.gender_combo.findText(self.patient_data.get("gender", ""))
        if idx >= 0: self.gender_combo.setCurrentIndex(idx)
        self.mobile_input.setText(self.patient_data.get("mobile_number", ""))
        self.address_input.setPlainText(self.patient_data.get("address", ""))
        self.history_input.setPlainText(self.patient_data.get("medical_history", ""))
        idx = self.diabetes_combo.findText(self.patient_data.get("diabetes_status", ""))
        if idx >= 0: self.diabetes_combo.setCurrentIndex(idx)
        idx = self.bp_combo.findText(self.patient_data.get("blood_pressure_status", ""))
        if idx >= 0: self.bp_combo.setCurrentIndex(idx)

    def _save(self):
        name = self.name_input.text().strip()
        if not name:
            QMessageBox.warning(self, "Validation", "Patient name is required")
            return
        age_text = self.age_input.text().strip()
        age = int(age_text) if age_text.isdigit() else None
        self.result_data = {
            "name": name,
            "age": age,
            "gender": self.gender_combo.currentText() or None,
            "mobile_number": self.mobile_input.text().strip() or None,
            "address": self.address_input.toPlainText().strip() or None,
            "medical_history": self.history_input.toPlainText().strip() or None,
            "diabetes_status": self.diabetes_combo.currentText() or None,
            "blood_pressure_status": self.bp_combo.currentText() or None,
        }
        self.accept()


class PatientView(QWidget):
    patient_selected = Signal(str)

    def __init__(self, doctor_id: str = None, parent=None):
        super().__init__(parent)
        self.doctor_id = doctor_id
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(16)

        header_layout = QHBoxLayout()
        title = QLabel("Patients")
        title.setStyleSheet("font-size: 24px; font-weight: 700; color: #E0E6ED;")
        header_layout.addWidget(title)
        header_layout.addStretch()

        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search by name, ID, or mobile...")
        self.search_input.setFixedWidth(250)
        self.search_input.textChanged.connect(self._search)
        header_layout.addWidget(self.search_input)

        add_btn = QPushButton("+ Add Patient")
        add_btn.setStyleSheet("background-color: #0A84FF; color: white; font-weight: 600; padding: 8px 20px;")
        add_btn.clicked.connect(self._add_patient)
        header_layout.addWidget(add_btn)

        layout.addLayout(header_layout)

        self.table = QTableWidget()
        self.table.setColumnCount(9)
        self.table.setHorizontalHeaderLabels([
            "Patient ID", "Name", "Age", "Gender", "Mobile", "Diabetes", "BP Status", "Scans", "Actions"
        ])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setVisible(False)
        layout.addWidget(self.table)

    def set_doctor(self, doctor_id: str):
        self.doctor_id = doctor_id
        self._load_patients()

    def _load_patients(self, search_query: str = None):
        if not self.doctor_id:
            return
        session = get_session()
        try:
            service = PatientService(session)
            if search_query:
                patients = service.search_patients(self.doctor_id, search_query)
            else:
                patients = service.get_patients_by_doctor(self.doctor_id)
            self.table.setRowCount(len(patients))
            for i, p in enumerate(patients):
                self.table.setItem(i, 0, QTableWidgetItem(p.patient_id))
                self.table.setItem(i, 1, QTableWidgetItem(p.name))
                self.table.setItem(i, 2, QTableWidgetItem(str(p.age) if p.age else ""))
                self.table.setItem(i, 3, QTableWidgetItem(p.gender or ""))
                self.table.setItem(i, 4, QTableWidgetItem(p.mobile_number or ""))
                self.table.setItem(i, 5, QTableWidgetItem(p.diabetes_status or ""))
                self.table.setItem(i, 6, QTableWidgetItem(p.blood_pressure_status or ""))
                self.table.setItem(i, 7, QTableWidgetItem(str(len(p.scan_records)) if p.scan_records else "0"))
                actions_widget = QWidget()
                actions_layout = QHBoxLayout(actions_widget)
                actions_layout.setContentsMargins(4, 2, 4, 2)
                actions_layout.setSpacing(4)
                view_btn = QPushButton("View")
                view_btn.setFixedSize(50, 24)
                view_btn.setStyleSheet("font-size: 10px; background-color: #0A84FF;")
                view_btn.clicked.connect(lambda checked, pid=p.id: self._view_patient(pid))
                actions_layout.addWidget(view_btn)
                delete_btn = QPushButton("Del")
                delete_btn.setFixedSize(40, 24)
                delete_btn.setStyleSheet("font-size: 10px; background-color: #FF4757;")
                delete_btn.clicked.connect(lambda checked, pid=p.id: self._delete_patient(pid))
                actions_layout.addWidget(delete_btn)
                self.table.setCellWidget(i, 8, actions_widget)
        finally:
            session.close()

    def _search(self, text: str):
        if len(text) >= 2:
            self._load_patients(text)
        elif not text:
            self._load_patients()

    def _add_patient(self):
        dialog = PatientFormDialog(self.doctor_id)
        if dialog.exec():
            session = get_session()
            try:
                service = PatientService(session)
                patient, error = service.create_patient(self.doctor_id, dialog.result_data)
                if error:
                    QMessageBox.warning(self, "Error", error)
                else:
                    self._load_patients()
            finally:
                session.close()

    def _view_patient(self, patient_id: str):
        self.patient_selected.emit(patient_id)

    def _delete_patient(self, patient_id: str):
        reply = QMessageBox.question(self, "Confirm", "Delete this patient and all records?",
                                     QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            session = get_session()
            try:
                service = PatientService(session)
                service.delete_patient(patient_id)
                self._load_patients()
            finally:
                session.close()

    def refresh(self):
        self._load_patients()
