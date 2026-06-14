from PySide6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QStackedWidget, QMessageBox, QApplication
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QIcon

from app.config import APP_NAME, APP_VERSION, DARK_BG
from app.database.connection import DatabaseManager, get_session
from app.database.seed import seed_default_doctor
from app.services.auth_service import AuthService
from app.ai.engine import ai_engine
from app.ui.components.sidebar import Sidebar
from app.ui.dialogs.model_loading_dialog import ModelLoadingDialog
from app.ui.components.topbar import TopBar
from app.ui.components.theme import DARK_STYLE
from app.ui.dialogs.login_dialog import LoginDialog
from app.ui.views.dashboard_view import DashboardView
from app.ui.views.patient_view import PatientView
from app.ui.views.doctor_view import DoctorView
from app.ui.views.scan_view import ScanView
from app.ui.views.reports_view import ReportsView
from app.ui.views.settings_view import SettingsView
from app.ui.views.logs_view import LogsView
from app.ui.views.ai_training_view import AITrainingView


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.doctor_info = None
        self.setWindowTitle(f"{APP_NAME} v{APP_VERSION}")
        self.setMinimumSize(1280, 800)
        self.resize(1440, 900)
        self.setStyleSheet(DARK_STYLE)

        self._init_database()
        self._show_login()

    def _init_database(self):
        try:
            db = DatabaseManager()
            db.create_tables()
            seed_default_doctor()
        except Exception as e:
            QMessageBox.critical(None, "Database Error",
                                 f"Failed to initialize database:\n{str(e)}\n\n"
                                 "The application will continue but database features may not work.")

    def _init_ai_models(self):
        try:
            dialog = ModelLoadingDialog(self)
            dialog.show()
            QApplication.processEvents()

            def progress_callback(pct, status, model_name="", device=""):
                dialog.update_progress(pct, status, model_name, device)

            result = ai_engine.load_models(progress_callback=progress_callback)

            if result.get("loaded"):
                dialog.mark_complete()
                msg = f"AI models loaded: {result['models_loaded']}/{result['models_attempted']} on {result['device']}"
            else:
                dialog.status_label.setText("Model loading had issues")
                dialog.model_label.setText(f"Errors: {len(result.get('errors', []))}")
                msg = f"AI model loading issues: {result.get('errors', [])}"

            print(msg)
            QApplication.processEvents()
            import time
            time.sleep(0.5)
            dialog.accept()
            return result
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Model loading error: {e}")
            try:
                dialog.accept()
            except Exception:
                pass
            return {"loaded": False, "models_loaded": 0, "models_attempted": 4, "errors": [str(e)]}

    def _show_login(self):
        self.login_dialog = LoginDialog(self)
        self.login_dialog.login_success.connect(self._on_login_success)
        result = self.login_dialog.exec()
        if result != LoginDialog.Accepted:
            QApplication.quit()
        if not hasattr(self, 'doctor_info') or self.doctor_info is None:
            QApplication.quit()

    def _on_login_success(self, doctor_info: dict):
        self.doctor_info = doctor_info
        try:
            model_result = self._init_ai_models()
            self._setup_ui()
            if model_result.get("loaded"):
                self.statusBar().showMessage(
                    f"AI Models: {model_result['models_loaded']} loaded on {model_result['device']}", 5000
                )
        except Exception as e:
            import traceback
            traceback.print_exc()
            # Still set up UI so the window is usable
            self._setup_ui()
            self.statusBar().showMessage(f"AI engine error: {e}", 10000)

    def _setup_ui(self):
        central = QWidget()
        central.setObjectName("centralWidget")
        self.setCentralWidget(central)

        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        self.sidebar = Sidebar()
        self.sidebar.page_changed.connect(self._change_page)
        main_layout.addWidget(self.sidebar)

        right_area = QWidget()
        right_layout = QVBoxLayout(right_area)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)

        doc_name = self.doctor_info.get("full_name", "Doctor")
        hospital = self.doctor_info.get("hospital_name", "Hospital")
        self.topbar = TopBar(doctor_name=doc_name, hospital_name=hospital)
        self.topbar.logout_clicked.connect(self._logout)
        self.topbar.search_requested.connect(self._on_search)
        right_layout.addWidget(self.topbar)

        self.stacked_widget = QStackedWidget()
        right_layout.addWidget(self.stacked_widget, 1)

        main_layout.addWidget(right_area, 1)

        self._init_pages()
        self._change_page(0)

    def _init_pages(self):
        doctor_id = self.doctor_info.get("doctor_id")

        self.dashboard_view = DashboardView(doctor_id)
        self.stacked_widget.addWidget(self.dashboard_view)

        self.patient_view = PatientView(doctor_id)
        self.stacked_widget.addWidget(self.patient_view)

        self.doctor_view = DoctorView()
        self.stacked_widget.addWidget(self.doctor_view)

        self.scan_view = ScanView(doctor_id)
        self.stacked_widget.addWidget(self.scan_view)

        self.reports_view = ReportsView(doctor_id)
        self.stacked_widget.addWidget(self.reports_view)

        self.ai_training_view = AITrainingView()
        self.stacked_widget.addWidget(self.ai_training_view)

        self.settings_view = SettingsView(doctor_id)
        self.stacked_widget.addWidget(self.settings_view)

        self.logs_view = LogsView()
        self.stacked_widget.addWidget(self.logs_view)

    def _change_page(self, index: int):
        self.stacked_widget.setCurrentIndex(index)
        self.sidebar.set_active(index)

        current_widget = self.stacked_widget.currentWidget()
        if hasattr(current_widget, 'set_doctor') and self.doctor_info:
            if index == 0:
                current_widget.set_doctor(self.doctor_info.get("doctor_id"))
            elif index == 1:
                current_widget.set_doctor(self.doctor_info.get("doctor_id"))
            elif index == 3:
                current_widget.set_doctor(self.doctor_info.get("doctor_id"))
            elif index == 4:
                current_widget.set_doctor(self.doctor_info.get("doctor_id"))
            elif index == 6:
                current_widget.set_doctor(self.doctor_info.get("doctor_id"))

        if hasattr(current_widget, 'refresh'):
            current_widget.refresh()
        if hasattr(current_widget, '_load_reports'):
            current_widget._load_reports()

    def _on_search(self, query: str):
        self._change_page(1)
        if hasattr(self.patient_view, 'search_input'):
            self.patient_view.search_input.setText(query)

    def _logout(self):
        reply = QMessageBox.question(self, "Logout", "Are you sure you want to logout?",
                                     QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.doctor_info = None
            self.hide()
            self._show_login()
