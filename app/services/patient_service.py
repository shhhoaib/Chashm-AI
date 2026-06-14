import random
import string
from sqlalchemy.orm import Session
from app.database.models import Patient


def generate_patient_id():
    prefix = "CHM"
    nums = ''.join(random.choices(string.digits, k=6))
    return f"{prefix}{nums}"


class PatientService:
    def __init__(self, session: Session):
        self.session = session

    def create_patient(self, doctor_id: str, data: dict) -> tuple:
        patient_id = generate_patient_id()
        while self.session.query(Patient).filter_by(patient_id=patient_id).first():
            patient_id = generate_patient_id()

        patient = Patient(
            patient_id=patient_id,
            doctor_id=doctor_id,
            name=data["name"],
            age=data.get("age"),
            gender=data.get("gender"),
            mobile_number=data.get("mobile_number"),
            address=data.get("address"),
            medical_history=data.get("medical_history"),
            diabetes_status=data.get("diabetes_status"),
            blood_pressure_status=data.get("blood_pressure_status"),
        )
        self.session.add(patient)
        self.session.commit()
        return patient, None

    def get_patient(self, patient_id: str):
        return self.session.query(Patient).filter_by(patient_id=patient_id).first()

    def get_patient_by_id(self, id: str):
        return self.session.query(Patient).get(id)

    def get_patients_by_doctor(self, doctor_id: str, limit: int = 100):
        return self.session.query(Patient).filter_by(doctor_id=doctor_id).order_by(
            Patient.created_at.desc()
        ).limit(limit).all()

    def search_patients(self, doctor_id: str, query: str):
        q = f"%{query}%"
        return self.session.query(Patient).filter(
            Patient.doctor_id == doctor_id,
            (Patient.name.ilike(q)) | (Patient.patient_id.ilike(q)) | (Patient.mobile_number.ilike(q))
        ).order_by(Patient.created_at.desc()).limit(50).all()

    def update_patient(self, patient_id: str, data: dict) -> tuple:
        patient = self.session.query(Patient).get(patient_id)
        if not patient:
            return None, "Patient not found"
        for key, value in data.items():
            if hasattr(patient, key) and value is not None:
                setattr(patient, key, value)
        self.session.commit()
        return patient, None

    def delete_patient(self, patient_id: str) -> bool:
        patient = self.session.query(Patient).get(patient_id)
        if not patient:
            return False
        self.session.delete(patient)
        self.session.commit()
        return True

    def get_patient_count(self, doctor_id: str = None) -> int:
        q = self.session.query(Patient)
        if doctor_id:
            q = q.filter_by(doctor_id=doctor_id)
        return q.count()
