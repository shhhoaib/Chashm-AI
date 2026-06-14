import os
import json
import shutil
import subprocess
from pathlib import Path
from datetime import datetime
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QLineEdit,
    QTableWidget, QTableWidgetItem, QHeaderView, QFileDialog, QMessageBox,
    QComboBox, QGroupBox, QTextEdit, QSplitter, QFrame, QProgressBar,
    QTabWidget, QGridLayout, QSizePolicy, QScrollArea
)
from PySide6.QtCore import Qt, QThread, Signal, QTimer
from PySide6.QtGui import QPixmap

from app.database.connection import get_session
from app.database.models import Patient, ScanRecord
from app.services.scan_service import ScanService
from app.services.patient_service import PatientService
from app.services.report_service import ReportService, generate_report_number
from app.ai.engine import ai_engine
from app.reporting.pdf_generator import generate_report, generate_qr_code, _get_patient_explanation
from app.config import BASE_DIR, VIZ_DIR


class AnalysisThread(QThread):
    finished = Signal(dict)
    error = Signal(str)
    progress = Signal(int)

    def __init__(self, image_path: str, scan_id: str, patient_id: str, doctor_id: str):
        super().__init__()
        self.image_path = image_path
        self.scan_id = scan_id
        self.patient_id = patient_id
        self.doctor_id = doctor_id

    def run(self):
        try:
            self.progress.emit(10)
            results = ai_engine.analyze(self.image_path)
            self.progress.emit(50)

            if results.get("error"):
                self.progress.emit(100)
                self.finished.emit({
                    "scan_id": self.scan_id,
                    "patient_id": self.patient_id,
                    "doctor_id": self.doctor_id,
                    "error": results["error"],
                })
                return

            self.progress.emit(70)
            viz_dir = VIZ_DIR / self.scan_id
            viz_results = ai_engine.generate_visualizations(self.image_path, results, str(viz_dir))

            self.progress.emit(90)
            session = get_session()
            try:
                scan_service = ScanService(session)
                update_data = {
                    "quality_score": results["quality"]["score"],
                    "quality_passed": 1 if results["quality"]["passed"] else 0,
                    "disease_detected": results["disease"]["disease"] if results.get("disease") else None,
                    "confidence_score": results.get("confidence_score", 0),
                    "severity_level": results.get("severity_level", "Normal"),
                    "severity_grade": results.get("severity_grade", 0),
                    "risk_level": results.get("risk_level", "Low"),
                    "affected_areas": results.get("affected_areas", ""),
                    "ai_findings": results.get("ai_findings", ""),
                    "ensemble_scores": results.get("ensemble_scores"),
                    "image_type": results.get("image_type", "unknown"),
                    "recommendation": results.get("recommendation", ""),
                    "disclaimer": results.get("disclaimer", ""),
                    "results_json": json.dumps({"findings": results.get("findings", [])}),
                    "heatmap_path": viz_results.get("heatmap_path"),
                    "annotated_path": viz_results.get("annotated_path"),
                    "segmentation_path": viz_results.get("segmentation_path"),
                    "grad_cam_path": viz_results.get("grad_cam_path"),
                }
                scan_service.update_scan_results(self.scan_id, update_data)
            finally:
                session.close()

            self.progress.emit(100)
            self.finished.emit({
                "scan_id": self.scan_id,
                "patient_id": self.patient_id,
                "doctor_id": self.doctor_id,
                "results": results,
                "viz": viz_results,
            })
        except Exception as e:
            self.error.emit(str(e))


class ScanView(QWidget):
    def __init__(self, doctor_id: str = None, parent=None):
        super().__init__(parent)
        self.doctor_id = doctor_id
        self.current_patient_id = None
        self.current_scan_id = None
        self.analysis_thread = None
        self.last_pdf_path = None
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(16)

        header = QLabel("Scan Analysis")
        header.setStyleSheet("font-size: 24px; font-weight: 700; color: #E0E6ED;")
        layout.addWidget(header)

        content = QHBoxLayout()
        content.setSpacing(16)

        # Left panel
        left_panel = QWidget()
        left_panel.setFixedWidth(350)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(12)

        # Patient selection
        patient_group = QGroupBox("Patient Selection")
        patient_grp_layout = QVBoxLayout(patient_group)
        self.patient_combo = QComboBox()
        self.patient_combo.setPlaceholderText("Select patient...")
        self.patient_combo.currentIndexChanged.connect(self._on_patient_selected)
        refresh_patients_btn = QPushButton("Refresh")
        refresh_patients_btn.setFixedHeight(28)
        refresh_patients_btn.setStyleSheet("font-size: 11px;")
        refresh_patients_btn.clicked.connect(self._load_patients)

        combo_layout = QHBoxLayout()
        combo_layout.addWidget(self.patient_combo, 1)
        combo_layout.addWidget(refresh_patients_btn)
        patient_grp_layout.addLayout(combo_layout)

        self.selected_patient_label = QLabel("No patient selected")
        self.selected_patient_label.setStyleSheet("color: #8892A0; font-size: 11px;")
        patient_grp_layout.addWidget(self.selected_patient_label)
        left_layout.addWidget(patient_group)

        # Upload
        upload_group = QGroupBox("Upload Scan")
        upload_layout = QVBoxLayout(upload_group)
        self.upload_btn = QPushButton("Select Image File")
        self.upload_btn.clicked.connect(self._upload_image)
        self.upload_btn.setStyleSheet("background-color: #0A84FF; color: white; padding: 12px; font-size: 14px;")
        upload_layout.addWidget(self.upload_btn)

        self.image_info_label = QLabel("No image selected")
        self.image_info_label.setStyleSheet("color: #8892A0; font-size: 11px;")
        upload_layout.addWidget(self.image_info_label)

        self.analyze_btn = QPushButton("Run AI Analysis")
        self.analyze_btn.setEnabled(False)
        self.analyze_btn.setStyleSheet("background-color: #2ED573; color: white; padding: 12px; font-size: 14px; font-weight: 600;")
        self.analyze_btn.clicked.connect(self._run_analysis)
        upload_layout.addWidget(self.analyze_btn)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        upload_layout.addWidget(self.progress_bar)
        left_layout.addWidget(upload_group)

        # Recent scans
        recent_group = QGroupBox("Recent Scans")
        recent_layout = QVBoxLayout(recent_group)
        self.recent_scans_table = QTableWidget()
        self.recent_scans_table.setColumnCount(3)
        self.recent_scans_table.setHorizontalHeaderLabels(["Patient", "Disease", "Date"])
        self.recent_scans_table.horizontalHeader().setStretchLastSection(True)
        self.recent_scans_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.recent_scans_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.recent_scans_table.verticalHeader().setVisible(False)
        self.recent_scans_table.setMaximumHeight(200)
        recent_layout.addWidget(self.recent_scans_table)
        left_layout.addWidget(recent_group)

        content.addWidget(left_panel)

        # Right panel - results
        right_panel = QScrollArea()
        right_panel.setWidgetResizable(True)
        right_content = QWidget()
        right_layout = QVBoxLayout(right_content)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(12)

        results_group = QGroupBox("Analysis Results")
        results_layout = QVBoxLayout(results_group)

        self.results_label = QLabel("Upload a scan and run analysis to see results")
        self.results_label.setStyleSheet("color: #8892A0; font-size: 13px; font-style: italic;")
        results_layout.addWidget(self.results_label)

        self.results_detail = QTextEdit()
        self.results_detail.setReadOnly(True)
        self.results_detail.setMaximumHeight(150)
        self.results_detail.setStyleSheet("background-color: #1B2838; border: 1px solid #2A3A4A; border-radius: 6px; padding: 8px; color: #E0E6ED;")
        results_layout.addWidget(self.results_detail)

        self.patient_explanation_label = QLabel("")
        self.patient_explanation_label.setVisible(False)
        self.patient_explanation_label.setWordWrap(True)
        self.patient_explanation_label.setStyleSheet(
            "color: #B0C4D9; font-size: 11px; background-color: #1B2838; "
            "border: 1px solid #2A3A4A; border-radius: 6px; padding: 10px; line-height: 1.4;"
        )
        results_layout.addWidget(self.patient_explanation_label)

        self.generate_report_btn = QPushButton("Generate PDF Report")
        self.generate_report_btn.setVisible(False)
        self.generate_report_btn.setStyleSheet("background-color: #0A84FF; color: white; font-weight: 600; padding: 10px;")
        self.generate_report_btn.clicked.connect(self._generate_report)
        results_layout.addWidget(self.generate_report_btn)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)

        self.save_report_btn = QPushButton("Save Report As...")
        self.save_report_btn.setVisible(False)
        self.save_report_btn.setStyleSheet("background-color: #10B981; color: white; font-weight: 600; padding: 10px;")
        self.save_report_btn.clicked.connect(self._save_report_as)
        btn_row.addWidget(self.save_report_btn)

        self.print_report_btn = QPushButton("Print Report")
        self.print_report_btn.setVisible(False)
        self.print_report_btn.setStyleSheet("background-color: #6B7280; color: white; font-weight: 600; padding: 10px;")
        self.print_report_btn.clicked.connect(self._print_report)
        btn_row.addWidget(self.print_report_btn)

        results_layout.addLayout(btn_row)

        right_layout.addWidget(results_group)

        # Visuals tab
        visuals_group = QGroupBox("Visual Analysis")
        visuals_layout = QVBoxLayout(visuals_group)

        self.visual_tabs = QTabWidget()
        self.original_label = QLabel("No image loaded")
        self.original_label.setAlignment(Qt.AlignCenter)
        self.original_label.setMinimumHeight(200)
        self.original_label.setStyleSheet("background-color: #1B2838; border-radius: 4px; color: #5A6A7A;")

        self.heatmap_label = QLabel("Run analysis to generate heatmap")
        self.heatmap_label.setAlignment(Qt.AlignCenter)
        self.heatmap_label.setMinimumHeight(200)
        self.heatmap_label.setStyleSheet("background-color: #1B2838; border-radius: 4px; color: #5A6A7A;")

        self.gradcam_label = QLabel("Run analysis to generate Grad-CAM")
        self.gradcam_label.setAlignment(Qt.AlignCenter)
        self.gradcam_label.setMinimumHeight(200)
        self.gradcam_label.setStyleSheet("background-color: #1B2838; border-radius: 4px; color: #5A6A7A;")

        self.annotated_label = QLabel("Run analysis to generate annotations")
        self.annotated_label.setAlignment(Qt.AlignCenter)
        self.annotated_label.setMinimumHeight(200)
        self.annotated_label.setStyleSheet("background-color: #1B2838; border-radius: 4px; color: #5A6A7A;")

        self.visual_tabs.addTab(self.original_label, "Original")
        self.visual_tabs.addTab(self.heatmap_label, "Heatmap")
        self.visual_tabs.addTab(self.gradcam_label, "Grad-CAM")
        self.visual_tabs.addTab(self.annotated_label, "Annotated")

        visuals_layout.addWidget(self.visual_tabs)
        right_layout.addWidget(visuals_group)

        right_panel.setWidget(right_content)
        content.addWidget(right_panel, 1)

        layout.addLayout(content)

    def set_doctor(self, doctor_id: str):
        self.doctor_id = doctor_id
        self._load_patients()
        self._load_recent_scans()

    def _load_patients(self):
        if not self.doctor_id:
            return
        session = get_session()
        try:
            service = PatientService(session)
            patients = service.get_patients_by_doctor(self.doctor_id)
            self.patient_combo.clear()
            self.patient_combo.addItem("-- Select Patient --", None)
            for p in patients:
                self.patient_combo.addItem(f"{p.name} ({p.patient_id})", p.id)
        finally:
            session.close()

    def _load_recent_scans(self):
        if not self.doctor_id:
            return
        session = get_session()
        try:
            service = ScanService(session)
            scans = service.get_recent_scans(self.doctor_id, limit=10)
            self.recent_scans_table.setRowCount(0)
            for row, scan in enumerate(scans):
                self.recent_scans_table.insertRow(row)
                patient = scan.patient
                name = patient.name if patient else "Unknown"
                disease = scan.disease_detected or "N/A"
                date = scan.created_at.strftime("%d %b") if scan.created_at else "N/A"
                self.recent_scans_table.setItem(row, 0, QTableWidgetItem(name))
                self.recent_scans_table.setItem(row, 1, QTableWidgetItem(disease))
                self.recent_scans_table.setItem(row, 2, QTableWidgetItem(date))
        finally:
            session.close()

    def _on_patient_selected(self, index):
        self.current_patient_id = self.patient_combo.currentData()
        text = self.patient_combo.currentText()
        if self.current_patient_id:
            self.selected_patient_label.setText(f"Selected: {text}")
        else:
            self.selected_patient_label.setText("No patient selected")
        has_img = hasattr(self, 'selected_image_path') and self.selected_image_path
        self.analyze_btn.setEnabled(bool(self.current_patient_id and self.doctor_id and has_img))

    def _upload_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Scan Image", "",
            "Image Files (*.png *.jpg *.jpeg *.bmp *.tiff *.dcm);;All Files (*.*)"
        )
        if file_path:
            self.selected_image_path = file_path
            self.image_info_label.setText(f"Selected: {Path(file_path).name}")
            self.analyze_btn.setEnabled(bool(self.current_patient_id and self.doctor_id))

    def _run_analysis(self):
        if not self.current_patient_id:
            QMessageBox.warning(self, "No Patient", "Please select a patient first")
            return
        if not self.selected_image_path:
            QMessageBox.warning(self, "No Image", "Please upload a scan image first")
            return

        scan_record = None
        session = get_session()
        try:
            service = ScanService(session)
            scan_record = service.create_scan(
                patient_id=self.current_patient_id,
                doctor_id=self.doctor_id,
                image_path=self.selected_image_path,
            )
            self.current_scan_id = scan_record.id
        finally:
            session.close()

        self.analyze_btn.setEnabled(False)
        self.analyze_btn.setText("Analyzing...")
        self.results_label.setText("AI analysis in progress...")
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.patient_explanation_label.setVisible(False)

        self.analysis_thread = AnalysisThread(
            image_path=self.selected_image_path,
            scan_id=self.current_scan_id,
            patient_id=self.current_patient_id,
            doctor_id=self.doctor_id,
        )
        self.analysis_thread.finished.connect(self._on_analysis_done)
        self.analysis_thread.error.connect(self._on_analysis_error)
        self.analysis_thread.progress.connect(self.progress_bar.setValue)
        self.analysis_thread.start()

    def _on_analysis_done(self, data: dict):
        self.analyze_btn.setEnabled(True)
        self.analyze_btn.setText("Run AI Analysis")
        self.progress_bar.setVisible(False)

        if data.get("error"):
            self.results_label.setText(f"Error: {data['error']}")
            self.results_detail.setPlainText(data['error'])
            self.generate_report_btn.setVisible(False)
            self.save_report_btn.setVisible(False)
            self.print_report_btn.setVisible(False)
            return

        results = data.get("results", {})
        disease = results.get("disease", {})
        viz = data.get("viz", {})

        if viz.get("heatmap_path"):
            pixmap = QPixmap(viz["heatmap_path"])
            if not pixmap.isNull():
                self.heatmap_label.setPixmap(pixmap.scaled(400, 300, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        if viz.get("annotated_path"):
            pixmap = QPixmap(viz["annotated_path"])
            if not pixmap.isNull():
                self.annotated_label.setPixmap(pixmap.scaled(400, 300, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        if viz.get("grad_cam_path"):
            pixmap = QPixmap(viz["grad_cam_path"])
            if not pixmap.isNull():
                self.gradcam_label.setPixmap(pixmap.scaled(400, 300, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        if viz.get("original_path"):
            pixmap = QPixmap(viz["original_path"])
            if not pixmap.isNull():
                self.original_label.setPixmap(pixmap.scaled(400, 300, Qt.KeepAspectRatio, Qt.SmoothTransformation))

        status = results.get("status", "success")
        image_type = results.get("image_type", "unknown")
        findings = results.get("findings", [])
        recommendation = results.get("recommendation", "")
        disclaimer = results.get("disclaimer", "")
        icd_10_code = results.get("icd_10_code", "")

        if status == "rejected":
            details = results.get("details", "Unrecognized image type.")
            self.results_label.setText("<b style='color:#FF4757'>Image Rejected</b>")
            parts = [
                f"<b>Reason:</b> {details}",
                f"<b>Image Type:</b> {image_type.replace('_', ' ').title() if image_type else 'Unknown'}",
            ]
            if recommendation:
                parts.append(f"<b>Recommendation:</b> {recommendation}")
            if disclaimer:
                parts.append(f"<br><small><i>{disclaimer}</i></small>")
            self.results_detail.setHtml("<br>".join(parts))
            self.generate_report_btn.setVisible(False)
            self.save_report_btn.setVisible(False)
            self.print_report_btn.setVisible(False)
            self.patient_explanation_label.setVisible(False)
            return

        if status == "low_quality":
            self.results_label.setText("<b style='color:#FFA502'>Insufficient Image Quality</b>")
            parts = [
                f"<b>Note:</b> The uploaded image does not meet the minimum quality requirements for reliable analysis.",
                f"<b>Image Type:</b> {image_type.replace('_', ' ').title() if image_type else 'Unknown'}",
            ]
            if recommendation:
                parts.append(f"<b>Recommendation:</b> {recommendation}")
            if disclaimer:
                parts.append(f"<br><small><i>{disclaimer}</i></small>")
            self.results_detail.setHtml("<br>".join(parts))
            self.generate_report_btn.setVisible(False)
            self.save_report_btn.setVisible(False)
            self.print_report_btn.setVisible(False)
            return

        if status == "low_confidence":
            self.results_label.setText("<b style='color:#FFA502'>Inconclusive Results</b>")
            parts = [
                f"<b>Image Type:</b> {image_type.replace('_', ' ').title() if image_type else 'Unknown'}",
            ]
            if findings:
                for f in findings:
                    parts.append(f"<b>{f.get('observation', 'Finding')}:</b> {f.get('details', '')}")
            if recommendation:
                parts.append(f"<b>Recommendation:</b> {recommendation}")
            if disclaimer:
                parts.append(f"<br><small><i>{disclaimer}</i></small>")
            self.results_detail.setHtml("<br>".join(parts))
            self.generate_report_btn.setVisible(False)
            self.save_report_btn.setVisible(False)
            self.print_report_btn.setVisible(False)
            self.patient_explanation_label.setVisible(False)
            return

        if status == "unsupported_type":
            self.results_label.setText("<b style='color:#FFA502'>Unsupported Image Type</b>")
            parts = [
                f"<b>Detected:</b> {image_type.replace('_', ' ').title() if image_type else 'Unknown'}",
                f"<b>Analysis:</b> The system cannot analyze this type of medical image.",
            ]
            if recommendation:
                parts.append(f"<b>Recommendation:</b> {recommendation}")
            if disclaimer:
                parts.append(f"<br><small><i>{disclaimer}</i></small>")
            self.results_detail.setHtml("<br>".join(parts))
            self.generate_report_btn.setVisible(False)
            self.save_report_btn.setVisible(False)
            self.print_report_btn.setVisible(False)
            self.patient_explanation_label.setVisible(False)
            return

        analysis_possible = results.get("analysis_possible", True)
        self.generate_report_btn.setVisible(True)
        can_report = status == "success" and analysis_possible
        self.save_report_btn.setVisible(self.last_pdf_path is not None)
        self.print_report_btn.setVisible(self.last_pdf_path is not None)

        if disease:
            disease_name = disease.get("disease", "Unknown")
            confidence = disease.get("confidence")
            severity = results.get("severity_level", "Unknown")
            severity_grade = results.get("severity_grade", 0)
            risk = results.get("risk_level", "Low")

            severity_colors = {"Critical": "#FF4757", "High": "#FF6B81", "Moderate": "#FFA502", "Low": "#2ED573", "Normal": "#2ED573"}
            color = severity_colors.get(severity, "#6B7280")
            try:
                confidence_str = f"(Confidence: {float(confidence):.1%})"
            except (ValueError, TypeError):
                confidence_str = "(Confidence: N/A)"

            self.results_label.setText(
                f"<b style='color:{color}'>" + "\u25cf " + f"</b> "
                f"<b>{disease_name}</b> "
                f"<span style='color:#8892A0; font-size: 11px;'>"
                f"{confidence_str}</span>"
            )

            detail_parts = []
            quality_score = results.get("quality_score")
            if quality_score is not None:
                try:
                    detail_parts.append(f"<b>Quality Score:</b> {float(quality_score):.1f}%")
                except (ValueError, TypeError):
                    detail_parts.append(f"<b>Quality Score:</b> {quality_score}")
            else:
                detail_parts.append("<b>Quality Score:</b> N/A")
            detail_parts.append(f"<b>Severity:</b> {severity} (Grade {severity_grade})")
            detail_parts.append(f"<b>Risk Level:</b> {risk}")
            if icd_10_code:
                detail_parts.append(f"<b>ICD-10:</b> {icd_10_code}")
            if findings:
                for f in findings:
                    obs = f.get("observation", "Finding")
                    det = f.get("details", "")
                    conf = f.get("confidence")
                    if conf is not None:
                        try:
                            conf_str = f" ({float(conf):.1%})"
                        except (ValueError, TypeError):
                            conf_str = f" ({conf})"
                    else:
                        conf_str = ""
                    detail_parts.append(f"<b>{obs}:</b> {det}{conf_str}")
            if recommendation:
                detail_parts.append(f"<b>Recommendation:</b> {recommendation}")
            if disclaimer:
                detail_parts.append(f"<br><small><i>{disclaimer}</i></small>")
            self.results_detail.setHtml("<br>".join(detail_parts) if detail_parts else disclaimer)

            explanation = _get_patient_explanation(disease_name)
            if explanation:
                self.patient_explanation_label.setText(
                    f"<b style='color:#33C3FF'>\u2139\ufe0f What This Means:</b><br><br>"
                    f"<span style='color:#B0C4D9; line-height: 1.4;'>{explanation}</span>"
                )
                self.patient_explanation_label.setVisible(True)
            else:
                self.patient_explanation_label.setVisible(False)
        else:
            self.results_label.setText("<b style='color:#2ED573'>No Abnormalities Detected</b>")
            detail_parts = []
            quality_score = results.get("quality_score")
            if quality_score is not None:
                try:
                    detail_parts.append(f"<b>Quality Score:</b> {float(quality_score):.1f}%")
                except (ValueError, TypeError):
                    detail_parts.append(f"<b>Quality Score:</b> {quality_score}")
            else:
                detail_parts.append("<b>Quality Score:</b> N/A")
            detail_parts.append("<b>Assessment:</b> No significant findings detected.")
            if recommendation:
                detail_parts.append(f"<b>Recommendation:</b> {recommendation}")
            if disclaimer:
                detail_parts.append(f"<br><small><i>{disclaimer}</i></small>")
            self.results_detail.setHtml("<br>".join(detail_parts) if detail_parts else disclaimer)

            image_type_lower = image_type.lower() if image_type else ""
            if "external" in image_type_lower:
                self.patient_explanation_label.setText(
                    "<b style='color:#33C3FF'>\u2139\ufe0f What This Means:</b><br><br>"
                    "<span style='color:#B0C4D9; line-height: 1.4;'>"
                    "No signs of eye disease were detected in this image. Your eyes appear healthy based "
                    "on this examination. Continue with regular eye check-ups and maintain good eye hygiene. "
                    "If you experience any symptoms like redness, pain, or vision changes, consult your "
                    "eye doctor promptly.</span>"
                )
            else:
                self.patient_explanation_label.setText(
                    "<b style='color:#33C3FF'>\u2139\ufe0f What This Means:</b><br><br>"
                    "<span style='color:#B0C4D9; line-height: 1.4;'>"
                    "No significant abnormalities were detected in this scan. Your retina appears healthy "
                    "based on the analysis. Continue with regular eye examinations and maintain a healthy "
                    "lifestyle to preserve your vision. If you notice any changes in your eyesight, "
                    "consult your eye doctor.</span>"
                )
            self.patient_explanation_label.setVisible(True)

    def _on_analysis_error(self, error_msg: str):
        self.analyze_btn.setEnabled(True)
        self.analyze_btn.setText("Run AI Analysis")
        self.progress_bar.setVisible(False)
        self.results_label.setText(f"Analysis failed: {error_msg}")
        self.results_detail.setPlainText(error_msg)
        self.generate_report_btn.setVisible(False)
        self.save_report_btn.setVisible(False)
        self.print_report_btn.setVisible(False)
        QMessageBox.warning(self, "Analysis Error", f"AI analysis failed: {error_msg}")

    def _generate_report(self):
        if not self.current_scan_id:
            QMessageBox.warning(self, "Error", "No scan selected")
            return

        session = get_session()
        try:
            scan = session.query(ScanRecord).get(self.current_scan_id)
            if not scan:
                QMessageBox.warning(self, "Error", "Scan not found")
                return

            from app.database.models import Doctor as DoctorModel
            doctor = session.query(DoctorModel).get(scan.doctor_id)
            patient = session.query(Patient).get(scan.patient_id)

            report_number = generate_report_number()
            qr_path = generate_qr_code(report_number)

            report_service = ReportService(session)
            clinical_summary = scan.ai_findings or self._build_clinical_summary(scan)
            follow_up_text = self._build_follow_up(scan)

            report = report_service.create_report(
                scan_id=self.current_scan_id,
                clinical_summary=clinical_summary,
                follow_up=follow_up_text,
                qr_path=qr_path,
            )

            stored_findings = []
            if scan.results_json:
                try:
                    stored = json.loads(scan.results_json)
                    stored_findings = stored.get("findings", [])
                except (json.JSONDecodeError, TypeError):
                    pass

            severity_grades = {
                "Normal": 0, "Mild": 1, "Moderate": 2, "Severe": 3, "Critical": 4,
                "Early": 1, "Advanced": 3, "Proliferative": 4, "End-Stage": 4,
            }

            disease_info = None
            if scan.disease_detected:
                sgrade = scan.severity_grade if scan.severity_grade is not None else severity_grades.get(scan.severity_level, 0)
                disease_info = {
                    "disease": scan.disease_detected,
                    "confidence": scan.confidence_score,
                    "severity": scan.severity_level or "Normal",
                    "severity_grade": sgrade,
                    "risk": scan.risk_level or "Low",
                }

            report_data = {
                "report_number": report.report_number,
                "doctor": {
                    "full_name": doctor.full_name if doctor else "Unknown",
                    "hospital_name": doctor.hospital_name if doctor else "Unknown",
                    "registration_number": doctor.registration_number if doctor else "N/A",
                    "qualification": doctor.qualification if doctor and doctor.qualification else "N/A",
                    "specialization": doctor.specialization if doctor and doctor.specialization else "N/A",
                    "mobile_number": doctor.mobile_number if doctor else "N/A",
                },
                "patient": {
                    "patient_id": patient.patient_id if patient else "N/A",
                    "name": patient.name if patient else "Unknown",
                    "age": patient.age if patient else "N/A",
                    "gender": patient.gender if patient else "N/A",
                    "mobile_number": patient.mobile_number if patient else "N/A",
                },
                "scan": {
                    "image_path": scan.image_path,
                    "image_type": scan.image_type or "unknown",
                    "disease": disease_info,
                    "confidence_score": scan.confidence_score or 0,
                    "severity_level": scan.severity_level or "Normal",
                    "risk_level": scan.risk_level or "Low",
                    "affected_areas": scan.affected_areas or "",
                    "ai_findings": scan.ai_findings or "",
                    "quality": {"score": scan.quality_score or 0},
                    "recommendation": scan.recommendation or "",
                    "disclaimer": scan.disclaimer or "",
                    "findings": stored_findings,
                    "ensemble_scores": json.loads(scan.ensemble_scores) if scan.ensemble_scores else None,
                    "heatmap_path": scan.heatmap_path,
                    "grad_cam_path": scan.grad_cam_path,
                    "annotated_path": scan.annotated_path,
                    "segmentation_path": scan.segmentation_path,
                },
                "qr_code_path": qr_path,
                "follow_up": follow_up_text,
                "clinical_summary": clinical_summary,
                "disease_name": scan.disease_detected or "N/A",
                "image_type": scan.image_type or "unknown",
            }

            pdf_path = generate_report(report_data)
            report.pdf_path = pdf_path
            session.commit()

            self.last_pdf_path = pdf_path
            self.save_report_btn.setVisible(True)
            self.print_report_btn.setVisible(True)

            QMessageBox.information(self, "Report Generated",
                                    f"PDF Report saved to:\n{pdf_path}")

            os.startfile(Path(pdf_path).parent)

        except Exception as e:
            session.rollback()
            QMessageBox.critical(self, "Error", f"Failed to generate report: {str(e)}")
        finally:
            session.close()

    def _build_clinical_summary(self, scan) -> str:
        parts = []
        if scan.image_type:
            parts.append(f"Image Type: {scan.image_type.replace('_', ' ').title()}")
        if scan.disease_detected:
            parts.append(f"Finding: {scan.disease_detected}")
        if scan.severity_level:
            parts.append(f"Severity: {scan.severity_level}")
        if scan.confidence_score is not None and scan.confidence_score > 0:
            parts.append(f"Confidence: {scan.confidence_score:.1f}%")
        if scan.ai_findings:
            parts.append(f"Analysis: {scan.ai_findings}")
        return " | ".join(parts) if parts else "AI analysis completed."

    def _build_follow_up(self, scan) -> str:
        sev = (scan.severity_level or "Normal").lower()
        if sev in ("critical", "severe", "high"):
            return ("URGENT: Immediate ophthalmology referral required within 24 hours. "
                    "Patient should seek emergency eye care immediately.")
        if sev in ("moderate", "medium"):
            return ("Schedule follow-up with an ophthalmologist within 1-2 weeks. "
                    "Close monitoring and further diagnostic evaluation recommended.")
        if sev in ("mild", "early", "low", "observed"):
            return ("Routine follow-up recommended in 3-6 months. "
                    "Continue regular eye care and monitor for any changes in vision.")
        return ("Follow-up recommended based on clinical findings. "
                "Please consult with a specialist for comprehensive evaluation.")

    def _save_report_as(self):
        if not self.last_pdf_path or not Path(self.last_pdf_path).exists():
            QMessageBox.warning(self, "No Report", "No report available. Generate a report first.")
            return

        src = Path(self.last_pdf_path)
        dst, _ = QFileDialog.getSaveFileName(
            self, "Save Report As",
            str(Path.home() / src.name),
            "PDF Files (*.pdf);;All Files (*.*)"
        )
        if not dst:
            return

        try:
            shutil.copy2(str(src), dst)
            QMessageBox.information(self, "Saved", f"Report saved to:\n{dst}")
            os.startfile(Path(dst).parent)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save report:\n{str(e)}")

    def _print_report(self):
        if not self.last_pdf_path or not Path(self.last_pdf_path).exists():
            QMessageBox.warning(self, "No Report", "No report available. Generate a report first.")
            return

        try:
            if os.name == "nt":
                os.startfile(self.last_pdf_path, "print")
            else:
                subprocess.Popen(["lp", self.last_pdf_path],
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            QMessageBox.information(self, "Print", "Report sent to printer.")
        except Exception as e:
            QMessageBox.critical(self, "Print Error",
                                 f"Could not print report:\n{str(e)}\n\n"
                                 f"You can open the file at:\n{self.last_pdf_path}")
