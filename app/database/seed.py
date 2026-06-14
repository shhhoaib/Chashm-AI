import bcrypt
from app.database.connection import get_session
from app.database.models import Doctor


def seed_default_doctor():
    session = get_session()
    try:
        existing = session.query(Doctor).filter_by(email="admin@chashm.ai").first()
        if existing:
            return
        password_hash = bcrypt.hashpw("admin123".encode(), bcrypt.gensalt()).decode()
        doctor = Doctor(
            full_name="Dr. Admin",
            hospital_name="Chashm AI Hospital",
            registration_number="CHASHM-001",
            mobile_number="+0000000000",
            email="admin@chashm.ai",
            qualification="MBBS, MS Ophthalmology",
            specialization="Ophthalmologist",
            password_hash=password_hash,
        )
        session.add(doctor)
        session.commit()
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()
