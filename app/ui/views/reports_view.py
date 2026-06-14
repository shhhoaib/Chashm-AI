import os
from pathlib import Path
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QLineEdit,
    QTableWidget, QTableWidgetItem, QHeaderView, QMessageBox, QFileDialog
)
from PySide6.QtCore import Qt

from app.database.connection import get_session
from app.database.models import Report, ScanRecord, Patient, Doctor
from app.services.report_service import ReportService
from app.config import REPORTS_DIR


class ReportsView(QWidget):
    def __init__(self, doctor_id: str = None, parent=None):
        super().__init__(parent)
        self.doctor_id = doctor_id
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(16)

        header_layout = QHBoxLayout()
        title = QLabel("Reports")
        title.setStyleSheet("font-size: 24px; font-weight: 700; color: #E0E6ED;")
        header_layout.addWidget(title)
        header_layout.addStretch()

        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self._load_reports)
        header_layout.addWidget(refresh_btn)

        open_folder_btn = QPushButton("Open Reports Folder")
        open_folder_btn.clicked.connect(self._open_reports_folder)
        header_layout.addWidget(open_folder_btn)

        layout.addLayout(header_layout)

        self.table = QTableWidget()
        self.table.setColumnCount(8)
        self.table.setHorizontalHeaderLabels([
            "Report #", "Patient", "Disease", "Confidence", "Severity",
            "Status", "Date", "Actions"
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
        self._load_reports()

    def _load_reports(self):
        session = get_session()
        try:
            reports = session.query(Report).order_by(Report.created_at.desc()).limit(100).all()
            self.table.setRowCount(len(reports))
            for i, r in enumerate(reports):
                scan = r.scan_record
                patient = scan.patient if scan else None

                self.table.setItem(i, 0, QTableWidgetItem(r.report_number))
                self.table.setItem(i, 1, QTableWidgetItem(patient.name if patient else "N/A"))
                self.table.setItem(i, 2, QTableWidgetItem(scan.disease_detected if scan else "N/A"))
                conf = f"{scan.confidence_score:.1f}%" if scan and scan.confidence_score else "N/A"
                self.table.setItem(i, 3, QTableWidgetItem(conf))
                self.table.setItem(i, 4, QTableWidgetItem(scan.severity_level if scan else "N/A"))
                self.table.setItem(i, 5, QTableWidgetItem("Verified" if r.is_verified else "Generated"))
                self.table.setItem(i, 6, QTableWidgetItem(
                    r.created_at.strftime("%d %b %Y %H:%M") if r.created_at else "N/A"
                ))

                actions_widget = QWidget()
                actions_layout = QHBoxLayout(actions_widget)
                actions_layout.setContentsMargins(4, 2, 4, 2)
                actions_layout.setSpacing(4)

                open_btn = QPushButton("Open PDF")
                open_btn.setFixedSize(70, 24)
                open_btn.setStyleSheet("font-size: 10px; background-color: #0A84FF;")
                open_btn.clicked.connect(lambda checked, path=r.pdf_path: self._open_pdf(path))
                actions_layout.addWidget(open_btn)

                verify_btn = QPushButton("Verify")
                verify_btn.setFixedSize(60, 24)
                verify_btn.setStyleSheet("font-size: 10px; background-color: #2ED573;")
                verify_btn.clicked.connect(lambda checked, rid=r.id: self._verify_report(rid))
                actions_layout.addWidget(verify_btn)

                self.table.setCellWidget(i, 7, actions_widget)
        finally:
            session.close()

    def _open_pdf(self, pdf_path: str):
        if pdf_path and Path(pdf_path).exists():
            os.startfile(pdf_path)
        else:
            QMessageBox.warning(self, "Not Found", "PDF file not found")

    def _verify_report(self, report_id: str):
        session = get_session()
        try:
            report_service = ReportService(session)
            report, error = report_service.verify_report(report_id)
            if error:
                QMessageBox.warning(self, "Error", error)
            else:
                QMessageBox.information(self, "Verified", "Report verified successfully")
                self._load_reports()
        finally:
            session.close()

    def _open_reports_folder(self):
        if REPORTS_DIR.exists():
            os.startfile(str(REPORTS_DIR))
