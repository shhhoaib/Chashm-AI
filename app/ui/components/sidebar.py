from PySide6.QtWidgets import QWidget, QVBoxLayout, QPushButton, QLabel, QButtonGroup, QSizePolicy
from PySide6.QtCore import Signal, Qt
from PySide6.QtGui import QFont, QPixmap
from pathlib import Path


class Sidebar(QWidget):
    page_changed = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(220)
        self.setObjectName("sidebar")
        self.setStyleSheet("""
            #sidebar {
                background-color: #00122B;
                border-right: 1px solid #001F54;
            }
        """)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        logo_container = QWidget()
        logo_container.setFixedHeight(80)
        logo_layout = QVBoxLayout(logo_container)
        logo_layout.setContentsMargins(16, 12, 16, 10)

        logo_path = Path(__file__).resolve().parent.parent.parent.parent / "logo.png"
        logo_label = QLabel()
        if logo_path.exists():
            pixmap = QPixmap(str(logo_path))
            logo_label.setPixmap(pixmap.scaled(36, 36, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            logo_label.setText("CH")
            logo_label.setStyleSheet("font-size: 22px; font-weight: 800; color: #33C3FF;")
        logo_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        logo_label.setFixedHeight(40)
        logo_layout.addWidget(logo_label)

        logo_sub = QLabel("Chashm AI")
        logo_sub.setStyleSheet("font-size: 9px; color: #8892A0; margin-top: -4px;")
        logo_sub.setAlignment(Qt.AlignLeft)
        logo_layout.addWidget(logo_sub)
        layout.addWidget(logo_container)

        self.btn_group = QButtonGroup(self)
        self.btn_group.setExclusive(True)

        nav_items = [
            ("  Dashboard", 0),
            ("  Patients", 1),
            ("  Doctors", 2),
            ("  Scan Analysis", 3),
            ("  Reports", 4),
            ("  AI Training", 5),
            ("  Settings", 6),
            ("  Logs", 7),
        ]

        nav_container = QWidget()
        nav_layout = QVBoxLayout(nav_container)
        nav_layout.setContentsMargins(0, 10, 0, 10)
        nav_layout.setSpacing(0)

        self.buttons = []
        for i, (text, idx) in enumerate(nav_items):
            btn = QPushButton(text)
            btn.setObjectName("sidebarBtn")
            btn.setCheckable(True)
            btn.setFixedHeight(44)
            btn_group_id = idx
            self.btn_group.addButton(btn, btn_group_id)
            if i == 0:
                btn.setChecked(True)
            self.buttons.append(btn)
            nav_layout.addWidget(btn)

        nav_layout.addStretch()
        layout.addWidget(nav_container)

        self.btn_group.idClicked.connect(self.page_changed.emit)

    def set_active(self, index: int):
        btn = self.btn_group.button(index)
        if btn:
            btn.setChecked(True)
