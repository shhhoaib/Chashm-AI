import uuid
from datetime import datetime
from sqlalchemy.orm import Session
from app.database.models import Report


def generate_report_number():
    date = datetime.now().strftime("%Y%m%d")
    uid = uuid.uuid4().hex[:6].upper()
    return f"CHM-RPT-{date}-{uid}"


class ReportService:
    def __init__(self, session: Session):
        self.session = session

    def create_report(self, scan_id: str, pdf_path: str = None, qr_path: str = None,
                      clinical_summary: str = None, follow_up: str = None) -> Report:
        report = Report(
            scan_id=scan_id,
            report_number=generate_report_number(),
            pdf_path=pdf_path,
            qr_code_path=qr_path,
            clinical_summary=clinical_summary,
            follow_up_recommendations=follow_up,
        )
        self.session.add(report)
        self.session.commit()
        return report

    def get_report(self, report_id: str):
        return self.session.query(Report).get(report_id)

    def get_report_by_scan(self, scan_id: str):
        return self.session.query(Report).filter_by(scan_id=scan_id).first()

    def get_recent_reports(self, limit: int = 20):
        return self.session.query(Report).order_by(Report.created_at.desc()).limit(limit).all()

    def verify_report(self, report_number: str) -> tuple:
        report = self.session.query(Report).filter_by(report_number=report_number).first()
        if not report:
            return None, "Report not found"
        report.is_verified = 1
        self.session.commit()
        return report, None
