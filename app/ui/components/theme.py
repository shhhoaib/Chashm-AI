from app.config import (
    PRIMARY_COLOR, SECONDARY_COLOR, ACCENT_COLOR, WHITE, DARK_BG, CARD_BG,
    TEXT_PRIMARY, TEXT_SECONDARY, SUCCESS_COLOR, WARNING_COLOR, DANGER_COLOR
)

DARK_STYLE = f"""
QWidget {{
    background-color: {DARK_BG};
    color: {TEXT_PRIMARY};
    font-family: 'Segoe UI', 'Arial', sans-serif;
    font-size: 13px;
}}
QMainWindow {{
    background-color: {DARK_BG};
}}
QPushButton {{
    background-color: {SECONDARY_COLOR};
    color: white;
    border: none;
    border-radius: 6px;
    padding: 8px 20px;
    font-size: 13px;
    font-weight: 600;
    min-height: 20px;
}}
QPushButton:hover {{
    background-color: {ACCENT_COLOR};
}}
QPushButton:pressed {{
    background-color: {PRIMARY_COLOR};
}}
QPushButton#dangerBtn {{
    background-color: {DANGER_COLOR};
}}
QPushButton#dangerBtn:hover {{
    background-color: #FF6B81;
}}
QPushButton#successBtn {{
    background-color: {SUCCESS_COLOR};
}}
QPushButton#successBtn:hover {{
    background-color: #7BED9F;
}}
QPushButton#sidebarBtn {{
    background-color: transparent;
    color: {TEXT_SECONDARY};
    text-align: left;
    padding: 12px 20px;
    border-radius: 0;
    font-weight: normal;
    font-size: 13px;
    min-height: 24px;
}}
QPushButton#sidebarBtn:hover {{
    background-color: rgba(10, 132, 255, 0.1);
    color: {WHITE};
}}
QPushButton#sidebarBtn:checked {{
    background-color: rgba(10, 132, 255, 0.2);
    color: {ACCENT_COLOR};
    border-left: 3px solid {ACCENT_COLOR};
}}
QLineEdit {{
    background-color: {CARD_BG};
    border: 1px solid #2A3A4A;
    border-radius: 6px;
    padding: 8px 12px;
    color: {TEXT_PRIMARY};
    font-size: 13px;
    min-height: 20px;
}}
QLineEdit:focus {{
    border-color: {SECONDARY_COLOR};
}}
QLineEdit::placeholder {{
    color: {TEXT_SECONDARY};
}}
QTextEdit {{
    background-color: {CARD_BG};
    border: 1px solid #2A3A4A;
    border-radius: 6px;
    padding: 8px;
    color: {TEXT_PRIMARY};
    font-size: 13px;
}}
QTextEdit:focus {{
    border-color: {SECONDARY_COLOR};
}}
QComboBox {{
    background-color: {CARD_BG};
    border: 1px solid #2A3A4A;
    border-radius: 6px;
    padding: 8px 12px;
    color: {TEXT_PRIMARY};
    font-size: 13px;
    min-height: 20px;
}}
QComboBox:focus {{
    border-color: {SECONDARY_COLOR};
}}
QComboBox::drop-down {{
    border: none;
    width: 30px;
}}
QComboBox::down-arrow {{
    image: none;
    border: none;
}}
QComboBox QAbstractItemView {{
    background-color: {CARD_BG};
    color: {TEXT_PRIMARY};
    selection-background-color: {SECONDARY_COLOR};
    border: 1px solid #2A3A4A;
}}
QLabel {{
    color: {TEXT_PRIMARY};
}}
QLabel#headerLabel {{
    font-size: 24px;
    font-weight: 700;
    color: {WHITE};
}}
QLabel#subheaderLabel {{
    font-size: 14px;
    color: {TEXT_SECONDARY};
}}
QLabel#statValue {{
    font-size: 28px;
    font-weight: 700;
    color: {WHITE};
}}
QLabel#statLabel {{
    font-size: 12px;
    color: {TEXT_SECONDARY};
}}
QLabel#sectionTitle {{
    font-size: 16px;
    font-weight: 600;
    color: {WHITE};
}}
QTableWidget {{
    background-color: {CARD_BG};
    border: 1px solid #2A3A4A;
    border-radius: 8px;
    gridline-color: #2A3A4A;
    color: {TEXT_PRIMARY};
    font-size: 12px;
}}
QTableWidget::item {{
    padding: 8px;
    border-bottom: 1px solid #2A3A4A;
}}
QTableWidget::item:selected {{
    background-color: rgba(10, 132, 255, 0.2);
    color: {WHITE};
}}
QHeaderView::section {{
    background-color: {PRIMARY_COLOR};
    color: {WHITE};
    padding: 10px 8px;
    border: none;
    font-weight: 600;
    font-size: 12px;
}}
QScrollBar:vertical {{
    background: {DARK_BG};
    width: 8px;
    border: none;
}}
QScrollBar::handle:vertical {{
    background: #2A3A4A;
    border-radius: 4px;
    min-height: 30px;
}}
QScrollBar::handle:vertical:hover {{
    background: {SECONDARY_COLOR};
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0;
}}
QScrollBar:horizontal {{
    background: {DARK_BG};
    height: 8px;
    border: none;
}}
QScrollBar::handle:horizontal {{
    background: #2A3A4A;
    border-radius: 4px;
    min-width: 30px;
}}
QScrollBar::handle:horizontal:hover {{
    background: {SECONDARY_COLOR};
}}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
    width: 0;
}}
QGroupBox {{
    background-color: {CARD_BG};
    border: 1px solid #2A3A4A;
    border-radius: 10px;
    margin-top: 12px;
    padding: 16px;
    font-weight: 600;
    font-size: 14px;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    padding: 0 10px;
    color: {WHITE};
}}
QProgressBar {{
    background-color: #2A3A4A;
    border: none;
    border-radius: 4px;
    height: 8px;
    text-align: center;
    font-size: 9px;
}}
QProgressBar::chunk {{
    background-color: {SECONDARY_COLOR};
    border-radius: 4px;
}}
QCheckBox {{
    spacing: 8px;
    color: {TEXT_PRIMARY};
}}
QCheckBox::indicator {{
    width: 18px;
    height: 18px;
    border-radius: 4px;
    border: 2px solid #2A3A4A;
    background-color: {CARD_BG};
}}
QCheckBox::indicator:checked {{
    background-color: {SECONDARY_COLOR};
    border-color: {SECONDARY_COLOR};
}}
QRadioButton {{
    spacing: 8px;
    color: {TEXT_PRIMARY};
}}
QRadioButton::indicator {{
    width: 18px;
    height: 18px;
    border-radius: 10px;
    border: 2px solid #2A3A4A;
    background-color: {CARD_BG};
}}
QRadioButton::indicator:checked {{
    background-color: {SECONDARY_COLOR};
    border-color: {SECONDARY_COLOR};
}}
QTabWidget::pane {{
    background-color: {DARK_BG};
    border: none;
}}
QTabBar::tab {{
    background-color: transparent;
    color: {TEXT_SECONDARY};
    padding: 10px 20px;
    border-bottom: 2px solid transparent;
    font-size: 13px;
}}
QTabBar::tab:selected {{
    color: {ACCENT_COLOR};
    border-bottom: 2px solid {ACCENT_COLOR};
}}
QTabBar::tab:hover {{
    color: {WHITE};
}}
QSplitter::handle {{
    background-color: #2A3A4A;
    width: 1px;
}}
QDialog {{
    background-color: {DARK_BG};
}}
QMessageBox {{
    background-color: {DARK_BG};
}}
QListWidget {{
    background-color: {CARD_BG};
    border: 1px solid #2A3A4A;
    border-radius: 8px;
    color: {TEXT_PRIMARY};
    font-size: 13px;
}}
QListWidget::item {{
    padding: 8px 12px;
    border-bottom: 1px solid #2A3A4A;
}}
QListWidget::item:selected {{
    background-color: rgba(10, 132, 255, 0.2);
    color: {WHITE};
}}
QListWidget::item:hover {{
    background-color: rgba(10, 132, 255, 0.1);
}}
"""
