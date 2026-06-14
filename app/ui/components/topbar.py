from PySide6.QtWidgets import QWidget, QHBoxLayout, QLabel, QLineEdit, QPushButton, QSizePolicy
from PySide6.QtCore import Signal, Qt
from PySide6.QtGui import QFont


class TopBar(QWidget):
    logout_clicked = Signal()
    search_requested = Signal(str)

    def __init__(self, doctor_name="Doctor", hospital_name="Hospital", parent=None):
        super().__init__(parent)
        self.setFixedHeight(60)
        self.setObjectName("topbar")
        self.setStyleSheet("""
            #topbar {
                background-color: #0D1B2A;
                border-bottom: 1px solid #001F54;
            }
        """)
        self.doctor_name = doctor_name
        self.hospital_name = hospital_name
        self._setup_ui()

    def _setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(20, 10, 20, 10)

        title = QLabel(f"{self.hospital_name}")
        title.setStyleSheet("font-size: 14px; font-weight: 600; color: #E0E6ED;")
        layout.addWidget(title)

        layout.addStretch()

        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search patients...")
        self.search_input.setFixedWidth(250)
        self.search_input.setStyleSheet("""
            QLineEdit {
                background-color: #1B2838;
                border: 1px solid #2A3A4A;
                border-radius: 15px;
                padding: 6px 14px;
                color: #E0E6ED;
                font-size: 12px;
            }
        """)
        self.search_input.returnPressed.connect(self._on_search)
        layout.addWidget(self.search_input)

        doc_label = QLabel(f"Dr. {self.doctor_name}")
        doc_label.setStyleSheet("font-size: 12px; color: #8892A0; padding: 0 10px;")
        layout.addWidget(doc_label)

        logout_btn = QPushButton("Logout")
        logout_btn.setFixedSize(80, 32)
        logout_btn.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                border: 1px solid #FF4757;
                color: #FF4757;
                border-radius: 6px;
                font-size: 12px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #FF4757;
                color: white;
            }
        """)
        logout_btn.clicked.connect(self.logout_clicked.emit)
        layout.addWidget(logout_btn)

    def _on_search(self):
        text = self.search_input.text().strip()
        self.search_requested.emit(text)

    def update_doctor_info(self, name: str, hospital: str):
        self.doctor_name = name
        self.hospital_name = hospital
        title = self.findChild(QLabel)
        if title:
            title.setText(hospital)
