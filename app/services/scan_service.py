import os
import json
import uuid
from datetime import datetime
from sqlalchemy.orm import Session
from app.database.models import ScanRecord
from app.config import SCANS_DIR, HEATMAPS_DIR, ANNOTATED_DIR, SEGMENTATIONS_DIR, GRADCAMS_DIR


for d in [SCANS_DIR, HEATMAPS_DIR, ANNOTATED_DIR, SEGMENTATIONS_DIR, GRADCAMS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


class ScanService:
    def __init__(self, session: Session):
        self.session = session

    def save_scan_image(self, image_data: bytes, filename: str) -> str:
        unique_name = f"{uuid.uuid4()}_{filename}"
        filepath = SCANS_DIR / unique_name
        with open(filepath, "wb") as f:
            f.write(image_data)
        return str(filepath)

    def create_scan(self, patient_id: str, doctor_id: str, image_path: str, filename: str = None) -> ScanRecord:
        ext = filename.split(".")[-1] if filename else "png"
        scan = ScanRecord(
            patient_id=patient_id,
            doctor_id=doctor_id,
            image_path=image_path,
            original_filename=filename,
            image_format=ext,
            status="uploaded",
        )
        self.session.add(scan)
        self.session.commit()
        return scan

    def get_scan(self, scan_id: str):
        return self.session.query(ScanRecord).get(scan_id)

    def get_scans_by_patient(self, patient_id: str):
        return self.session.query(ScanRecord).filter_by(patient_id=patient_id).order_by(
            ScanRecord.created_at.desc()
        ).all()

    def get_recent_scans(self, doctor_id: str, limit: int = 20):
        return self.session.query(ScanRecord).filter_by(doctor_id=doctor_id).order_by(
            ScanRecord.created_at.desc()
        ).limit(limit).all()

    def update_scan_results(self, scan_id: str, results: dict) -> ScanRecord:
        scan = self.session.query(ScanRecord).get(scan_id)
        if not scan:
            return None
        for key, value in results.items():
            if hasattr(scan, key) and value is not None:
                if key == "ensemble_scores" and isinstance(value, dict):
                    value = json.dumps(value)
                setattr(scan, key, value)
        result_status = results.get("status", "")
        if result_status in ("rejected", "low_quality", "low_confidence", "unsupported_type"):
            scan.status = "inconclusive"
        elif results.get("disease_detected"):
            scan.status = "completed"
        else:
            scan.status = "normal"
        self.session.commit()
        return scan

    def get_scan_count(self, doctor_id: str = None) -> int:
        q = self.session.query(ScanRecord)
        if doctor_id:
            q = q.filter_by(doctor_id=doctor_id)
        return q.count()

    def get_today_scan_count(self, doctor_id: str = None) -> int:
        today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        q = self.session.query(ScanRecord).filter(ScanRecord.created_at >= today_start)
        if doctor_id:
            q = q.filter_by(doctor_id=doctor_id)
        return q.count()

    def get_positive_case_count(self, doctor_id: str = None) -> int:
        q = self.session.query(ScanRecord).filter(
            ScanRecord.status == "completed",
            ScanRecord.disease_detected.isnot(None),
            ScanRecord.disease_detected != ""
        )
        if doctor_id:
            q = q.filter_by(doctor_id=doctor_id)
        return q.count()

    def get_pending_review_count(self, doctor_id: str = None) -> int:
        q = self.session.query(ScanRecord).filter_by(status="inconclusive")
        if doctor_id:
            q = q.filter_by(doctor_id=doctor_id)
        return q.count()
