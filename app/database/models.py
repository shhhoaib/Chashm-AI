import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, String, Text, Integer, Float, DateTime, ForeignKey, Enum, LargeBinary
from sqlalchemy.orm import DeclarativeBase, relationship
import enum

class Base(DeclarativeBase):
    pass

def generate_uuid():
    return str(uuid.uuid4())

class Doctor(Base):
    __tablename__ = "doctors"

    id = Column(String, primary_key=True, default=generate_uuid)
    full_name = Column(String(255), nullable=False)
    hospital_name = Column(String(255), nullable=False)
    registration_number = Column(String(100), unique=True, nullable=False)
    mobile_number = Column(String(20), nullable=False)
    email = Column(String(255), unique=True, nullable=False)
    qualification = Column(String(255))
    specialization = Column(String(255))
    hospital_logo_path = Column(String(500))
    password_hash = Column(String(255), nullable=False)
    is_active = Column(Integer, default=1)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    patients = relationship("Patient", back_populates="doctor")
    scan_records = relationship("ScanRecord", back_populates="doctor")

class Patient(Base):
    __tablename__ = "patients"

    id = Column(String, primary_key=True, default=generate_uuid)
    patient_id = Column(String(50), unique=True, nullable=False)
    name = Column(String(255), nullable=False)
    age = Column(Integer)
    gender = Column(String(10))
    mobile_number = Column(String(20))
    address = Column(Text)
    medical_history = Column(Text)
    diabetes_status = Column(String(50))
    blood_pressure_status = Column(String(50))
    doctor_id = Column(String, ForeignKey("doctors.id"))
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    doctor = relationship("Doctor", back_populates="patients")
    scan_records = relationship("ScanRecord", back_populates="patient")

class ScanRecord(Base):
    __tablename__ = "scan_records"

    id = Column(String, primary_key=True, default=generate_uuid)
    patient_id = Column(String, ForeignKey("patients.id"), nullable=False)
    doctor_id = Column(String, ForeignKey("doctors.id"), nullable=False)
    image_path = Column(String(500), nullable=False)
    original_filename = Column(String(255))
    image_format = Column(String(20))
    quality_score = Column(Float)
    quality_passed = Column(Integer, default=0)
    disease_detected = Column(String(255))
    confidence_score = Column(Float)
    severity_level = Column(String(50))
    severity_grade = Column(Integer)
    risk_level = Column(String(50))
    affected_areas = Column(Text)
    ai_findings = Column(Text)
    ensemble_scores = Column(Text)
    heatmap_path = Column(String(500))
    annotated_path = Column(String(500))
    segmentation_path = Column(String(500))
    grad_cam_path = Column(String(500))
    image_type = Column(String(50))
    recommendation = Column(Text)
    disclaimer = Column(Text)
    results_json = Column(Text)
    status = Column(String(50), default="pending")
    notes = Column(Text)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    patient = relationship("Patient", back_populates="scan_records")
    doctor = relationship("Doctor", back_populates="scan_records")
    report = relationship("Report", back_populates="scan_record", uselist=False)

class Report(Base):
    __tablename__ = "reports"

    id = Column(String, primary_key=True, default=generate_uuid)
    scan_id = Column(String, ForeignKey("scan_records.id"), unique=True, nullable=False)
    report_number = Column(String(50), unique=True, nullable=False)
    pdf_path = Column(String(500))
    qr_code_path = Column(String(500))
    clinical_summary = Column(Text)
    follow_up_recommendations = Column(Text)
    is_verified = Column(Integer, default=0)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    scan_record = relationship("ScanRecord", back_populates="report")

class AuditLog(Base):
    __tablename__ = "audit_logs"

    id = Column(String, primary_key=True, default=generate_uuid)
    doctor_id = Column(String, ForeignKey("doctors.id"))
    action = Column(String(255), nullable=False)
    details = Column(Text)
    ip_address = Column(String(50))
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
