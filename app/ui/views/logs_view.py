from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTableWidget, QTableWidgetItem, QHeaderView
)
from PySide6.QtCore import Qt
from app.database.connection import get_session
from app.database.models import AuditLog


class LogsView(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(16)

        header_layout = QHBoxLayout()
        title = QLabel("Audit Logs")
        title.setStyleSheet("font-size: 24px; font-weight: 700; color: #E0E6ED;")
        header_layout.addWidget(title)
        header_layout.addStretch()
        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self._load_logs)
        header_layout.addWidget(refresh_btn)
        layout.addLayout(header_layout)

        self.table = QTableWidget()
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(["Doctor ID", "Action", "Details", "IP Address", "Timestamp"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setVisible(False)
        layout.addWidget(self.table)

        self._load_logs()

    def _load_logs(self):
        session = get_session()
        try:
            logs = session.query(AuditLog).order_by(AuditLog.created_at.desc()).limit(100).all()
            self.table.setRowCount(len(logs))
            for i, log in enumerate(logs):
                self.table.setItem(i, 0, QTableWidgetItem(log.doctor_id or "System"))
                self.table.setItem(i, 1, QTableWidgetItem(log.action))
                self.table.setItem(i, 2, QTableWidgetItem(log.details or ""))
                self.table.setItem(i, 3, QTableWidgetItem(log.ip_address or ""))
                self.table.setItem(i, 4, QTableWidgetItem(
                    log.created_at.strftime("%d %b %Y %H:%M:%S") if log.created_at else "N/A"
                ))
        finally:
            session.close()
