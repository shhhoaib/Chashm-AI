from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QGridLayout,
    QTableWidget, QTableWidgetItem, QHeaderView, QFrame, QPushButton, QSizePolicy
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont
from datetime import datetime

from app.database.connection import get_session
from app.database.models import Doctor, Patient, ScanRecord, AuditLog
from app.services.scan_service import ScanService
from app.services.patient_service import PatientService
from app.ui.components.stat_card import StatCard


class DashboardView(QWidget):
    def __init__(self, doctor_id: str = None, parent=None):
        super().__init__(parent)
        self.doctor_id = doctor_id
        self._setup_ui()
        self._refresh_timer = QTimer(self)
        self._refresh_timer.timeout.connect(self.refresh_data)
        self._refresh_timer.start(10000)

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(16)

        header = QLabel("Dashboard")
        header.setStyleSheet("font-size: 24px; font-weight: 700; color: #E0E6ED;")
        layout.addWidget(header)

        sub = QLabel("Overview of your practice")
        sub.setStyleSheet("font-size: 13px; color: #8892A0; margin-top: -10px;")
        layout.addWidget(sub)

        # Stat cards
        stats_layout = QHBoxLayout()
        stats_layout.setSpacing(16)

        self.total_patients_card = StatCard("Total Patients", "--", "#0A84FF", "Registered patients")
        self.today_scans_card = StatCard("Today's Scans", "--", "#2ED573", "Scans performed today")
        self.detected_cases_card = StatCard("Detected Cases", "--", "#FFA502", "Positive findings")
        self.pending_review_card = StatCard("Pending Review", "--", "#FF4757", "Awaiting review")

        stats_layout.addWidget(self.total_patients_card)
        stats_layout.addWidget(self.today_scans_card)
        stats_layout.addWidget(self.detected_cases_card)
        stats_layout.addWidget(self.pending_review_card)
        layout.addLayout(stats_layout)

        # Recent scans
        recent_label = QLabel("Recent Scans")
        recent_label.setStyleSheet("font-size: 16px; font-weight: 600; color: #E0E6ED; margin-top: 8px;")
        layout.addWidget(recent_label)

        self.recent_table = QTableWidget()
        self.recent_table.setColumnCount(6)
        self.recent_table.setHorizontalHeaderLabels(["Patient", "Disease", "Confidence", "Severity", "Status", "Date"])
        self.recent_table.horizontalHeader().setStretchLastSection(True)
        self.recent_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.recent_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.recent_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.recent_table.setAlternatingRowColors(True)
        self.recent_table.verticalHeader().setVisible(False)
        self.recent_table.setMinimumHeight(200)
        layout.addWidget(self.recent_table)

    def set_doctor(self, doctor_id: str):
        self.doctor_id = doctor_id
        self.refresh_data()

    def refresh_data(self):
        if not self.doctor_id:
            return
        session = get_session()
        try:
            scan_service = ScanService(session)
            patient_service = PatientService(session)

            total = patient_service.get_patient_count(self.doctor_id)
            today = scan_service.get_today_scan_count(self.doctor_id)
            detected = scan_service.get_positive_case_count(self.doctor_id)
            pending = scan_service.get_pending_review_count(self.doctor_id)

            self.total_patients_card.set_value(str(total))
            self.today_scans_card.set_value(str(today))
            self.detected_cases_card.set_value(str(detected))
            self.pending_review_card.set_value(str(pending))

            recent = scan_service.get_recent_scans(self.doctor_id, 20)
            self.recent_table.setRowCount(len(recent))
            for i, scan in enumerate(recent):
                patient = scan.patient
                p_name = patient.name if patient else "Unknown"
                self.recent_table.setItem(i, 0, QTableWidgetItem(p_name))
                self.recent_table.setItem(i, 1, QTableWidgetItem(scan.disease_detected or "N/A"))
                conf = f"{scan.confidence_score:.1f}%" if scan.confidence_score else "N/A"
                self.recent_table.setItem(i, 2, QTableWidgetItem(conf))
                self.recent_table.setItem(i, 3, QTableWidgetItem(scan.severity_level or "N/A"))
                self.recent_table.setItem(i, 4, QTableWidgetItem(scan.status.capitalize() if scan.status else "Pending"))
                self.recent_table.setItem(i, 5, QTableWidgetItem(
                    scan.created_at.strftime("%d %b %Y %H:%M") if scan.created_at else "N/A"
                ))
        except Exception as e:
            print(f"Dashboard refresh error: {e}")
        finally:
            session.close()
