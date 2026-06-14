from PySide6.QtWidgets import QFrame, QVBoxLayout, QLabel
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont


class StatCard(QFrame):
    def __init__(self, title: str, value: str, color: str = "#0A84FF", subtitle: str = "", parent=None):
        super().__init__(parent)
        self.setObjectName("statCard")
        self.setStyleSheet(f"""
            #statCard {{
                background-color: #1B2838;
                border: 1px solid #2A3A4A;
                border-radius: 10px;
                border-top: 3px solid {color};
                padding: 16px;
            }}
            #statCard:hover {{
                border-color: {color};
                background-color: #1E2D40;
            }}
        """)
        self.setMinimumHeight(110)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(4)

        self.value_label = QLabel(str(value))
        self.value_label.setStyleSheet(f"font-size: 28px; font-weight: 700; color: {color};")
        self.value_label.setAlignment(Qt.AlignLeft)
        layout.addWidget(self.value_label)

        self.title_label = QLabel(title)
        self.title_label.setStyleSheet("font-size: 12px; color: #8892A0;")
        self.title_label.setAlignment(Qt.AlignLeft)
        layout.addWidget(self.title_label)

        if subtitle:
            self.sub_label = QLabel(subtitle)
            self.sub_label.setStyleSheet("font-size: 10px; color: #5A6A7A;")
            self.sub_label.setAlignment(Qt.AlignLeft)
            layout.addWidget(self.sub_label)

    def set_value(self, value: str):
        self.value_label.setText(str(value))
