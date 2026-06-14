from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QGroupBox, QProgressBar, QMessageBox, QTextEdit, QFrame
)
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QFont

from app.ai.engine import ai_engine
from app.ai.trainer import run_training
from app.config import SECONDARY_COLOR, SUCCESS_COLOR, DANGER_COLOR, WARNING_COLOR, CARD_BG, TRAINED_DIR


class TrainingThread(QThread):
    progress = Signal(int, str)
    finished = Signal(dict)
    error = Signal(str)

    def __init__(self, epochs=5, batch_size=8, learning_rate=0.001):
        super().__init__()
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate

    def run(self):
        try:
            def cb(pct, msg):
                self.progress.emit(pct, msg)
            result = run_training(
                epochs=self.epochs,
                batch_size=self.batch_size,
                lr=self.learning_rate,
                progress_callback=cb,
            )
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


class AITrainingView(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.training_thread = None
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(16)

        header = QLabel("AI Training")
        header.setStyleSheet("font-size: 24px; font-weight: 700; color: #E0E6ED;")
        layout.addWidget(header)

        sub = QLabel("Manage model training, fine-tuning, and evaluation")
        sub.setStyleSheet("font-size: 13px; color: #8892A0; margin-top: -8px;")
        layout.addWidget(sub)

        # Model Status
        status_group = QGroupBox("Model Status")
        self.status_layout = QVBoxLayout(status_group)
        self.status_layout.setSpacing(6)

        self.status_labels = {}
        model_keys = [
            ("type_classifier", "Type Classifier", "Image type (CNN)"),
            ("external_eye_model", "External Eye Model", "Disease classifier (CNN)"),
            ("efficientnet", "EfficientNetV2-S", "Fundus feature extraction"),
            ("convnext", "ConvNeXt Tiny", "Fundus feature extraction"),
            ("resnet", "ResNet18", "Fundus + Grad-CAM"),
            ("mobilenet", "MobileNetV3-Small", "Fundus feature extraction"),
        ]
        for key, display_name, purpose in model_keys:
            row = QFrame()
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(8, 4, 8, 4)

            name_label = QLabel(display_name)
            name_label.setStyleSheet("font-size: 12px; font-weight: 600; color: #E0E6ED;")
            name_label.setFixedWidth(180)
            row_layout.addWidget(name_label)

            status_label = QLabel("Checking...")
            status_label.setStyleSheet("font-size: 12px;")
            status_label.setFixedWidth(120)
            row_layout.addWidget(status_label)

            train_badge = QLabel(purpose)
            train_badge.setStyleSheet("font-size: 10px; color: #5A6A7A;")
            row_layout.addWidget(train_badge)

            row_layout.addStretch()
            self.status_layout.addWidget(row)
            self.status_labels[key] = status_label

        self.status_layout.addStretch()
        layout.addWidget(status_group)

        # Training Controls
        train_group = QGroupBox("Fine-Tune Models")
        train_layout = QVBoxLayout(train_group)

        info = QLabel(
            "Fine-tune pre-trained models on eye disease datasets.<br>"
            "Training uses synthetic data generated from the AI engine's knowledge base.<br>"
            "<i>For production, replace with real datasets (APTOS, ODIR, REFUGE, EyePACS).</i>"
        )
        info.setTextFormat(Qt.RichText)
        info.setStyleSheet("font-size: 11px; color: #8892A0;")
        train_layout.addWidget(info)

        controls = QHBoxLayout()
        self.train_btn = QPushButton("Start Fine-Tuning")
        self.train_btn.setStyleSheet(f"background-color: {SECONDARY_COLOR}; color: white; font-weight: 600; padding: 10px 24px;")
        self.train_btn.clicked.connect(self._start_training)
        controls.addWidget(self.train_btn)

        self.reset_btn = QPushButton("Reset to Base Weights")
        self.reset_btn.setStyleSheet(f"background-color: transparent; border: 1px solid {DANGER_COLOR}; color: {DANGER_COLOR};")
        self.reset_btn.clicked.connect(self._reset_weights)
        controls.addWidget(self.reset_btn)

        controls.addStretch()
        train_layout.addLayout(controls)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setStyleSheet(f"""
            QProgressBar {{
                background-color: #2A3A4A; border: none; border-radius: 6px;
                height: 20px; text-align: center; font-size: 11px; font-weight: 600; color: white;
            }}
            QProgressBar::chunk {{ background-color: {SECONDARY_COLOR}; border-radius: 6px; }}
        """)
        train_layout.addWidget(self.progress_bar)

        self.train_status = QLabel("")
        self.train_status.setStyleSheet("font-size: 11px; color: #8892A0;")
        train_layout.addWidget(self.train_status)

        layout.addWidget(train_group)

        # Logs
        logs_group = QGroupBox("Training Logs")
        logs_layout = QVBoxLayout(logs_group)
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setMaximumHeight(200)
        self.log_output.setStyleSheet(f"background-color: {CARD_BG}; color: #E0E6ED; font-size: 11px; font-family: Consolas; border: 1px solid #2A3A4A; border-radius: 6px; padding: 8px;")
        logs_layout.addWidget(self.log_output)
        layout.addWidget(logs_group)

        layout.addStretch()

        self._refresh_status()

    def _refresh_status(self):
        for key, label in self.status_labels.items():
            if key in ai_engine.models or getattr(ai_engine, key, None) is not None:
                if key == "type_classifier":
                    p = TRAINED_DIR / "type_classifier.pt"
                    if p.exists():
                        label.setText("Trained")
                        label.setStyleSheet(f"font-size: 12px; color: {SUCCESS_COLOR}; font-weight: 600;")
                    else:
                        label.setText("Not trained")
                        label.setStyleSheet(f"font-size: 12px; color: {WARNING_COLOR}; font-weight: 600;")
                elif key == "external_eye_model":
                    p = TRAINED_DIR / "external_eye_model.pt"
                    if p.exists():
                        label.setText("Trained")
                        label.setStyleSheet(f"font-size: 12px; color: {SUCCESS_COLOR}; font-weight: 600;")
                    else:
                        label.setText("Not trained")
                        label.setStyleSheet(f"font-size: 12px; color: {WARNING_COLOR}; font-weight: 600;")
                else:
                    trained_path = ai_engine._get_trained_path(key)
                    if trained_path and trained_path.exists():
                        label.setText("Trained")
                        label.setStyleSheet(f"font-size: 12px; color: {SUCCESS_COLOR}; font-weight: 600;")
                    else:
                        label.setText("Loaded (base)")
                        label.setStyleSheet(f"font-size: 12px; color: {WARNING_COLOR}; font-weight: 600;")
            else:
                label.setText("Not loaded")
                label.setStyleSheet(f"font-size: 12px; color: {DANGER_COLOR}; font-weight: 600;")

    def _start_training(self):
        if self.training_thread and self.training_thread.isRunning():
            QMessageBox.warning(self, "Training", "Training is already in progress")
            return

        self.train_btn.setEnabled(False)
        self.train_btn.setText("Training...")
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.log_output.clear()
        self.log_output.append("Starting fine-tuning...")

        self.training_thread = TrainingThread(epochs=5, batch_size=8, learning_rate=0.001)
        self.training_thread.progress.connect(self._on_training_progress)
        self.training_thread.finished.connect(self._on_training_finished)
        self.training_thread.error.connect(self._on_training_error)
        self.training_thread.start()

    def _on_training_progress(self, pct: int, msg: str):
        self.progress_bar.setValue(pct)
        self.train_status.setText(msg)
        self.log_output.append(msg)
        self.log_output.verticalScrollBar().setValue(
            self.log_output.verticalScrollBar().maximum()
        )

    def _on_training_finished(self, result: dict):
        self.train_btn.setEnabled(True)
        self.train_btn.setText("Start Fine-Tuning")
        self.progress_bar.setVisible(False)
        self._refresh_status()

        if result.get("success"):
            self.train_status.setText("Training complete!")
            self.log_output.append("\n" + "=" * 40)
            self.log_output.append("TRAINING COMPLETE")
            all_res = result.get("results", {})
            for phase, phase_res in all_res.items():
                self.log_output.append(f"\n--- {phase} ---")
                if isinstance(phase_res, dict):
                    for k, v in phase_res.items():
                        if isinstance(v, dict):
                            self.log_output.append(f"  {k}: loss={v.get('final_loss', '?'):.3f} acc={v.get('accuracy', '?'):.2f}")
                        else:
                            self.log_output.append(f"  {k}: {v}")
                else:
                    self.log_output.append(f"  {phase_res}")
            QMessageBox.information(self, "Training Complete",
                                    "All models trained successfully!\n"
                                    "Reload the app to use new weights.")
        else:
            self.train_status.setText("Training had issues")
            self.log_output.append(f"Errors: {result.get('errors', [])}")

    def _on_training_error(self, err: str):
        self.train_btn.setEnabled(True)
        self.train_btn.setText("Start Fine-Tuning")
        self.progress_bar.setVisible(False)
        self.train_status.setText(f"Error: {err}")
        self.log_output.append(f"ERROR: {err}")
        QMessageBox.critical(self, "Training Error", f"Training failed:\n{err}")

    def _reset_weights(self):
        reply = QMessageBox.question(
            self, "Reset Weights",
            "Reset all models to base weights?\nAll trained weights will be lost.",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            for key in list(ai_engine.models.keys()):
                ai_engine._remove_trained_weights(key)
            for extra in ["type_classifier.pt", "external_eye_model.pt"]:
                p = TRAINED_DIR / extra
                if p.exists():
                    p.unlink()
            self.log_output.append("All custom weights removed.")
            self._refresh_status()

    def refresh(self):
        self._refresh_status()
