import bcrypt
import jwt
from datetime import datetime, timedelta, timezone
from sqlalchemy.orm import Session
from app.database.models import Doctor
from app.config import SECRET_KEY, JWT_ALGORITHM, JWT_EXPIRATION_HOURS


class AuthService:
    def __init__(self, session: Session):
        self.session = session

    def register_doctor(self, data: dict) -> tuple:
        existing = self.session.query(Doctor).filter(
            (Doctor.email == data["email"]) |
            (Doctor.registration_number == data["registration_number"])
        ).first()
        if existing:
            return None, "Doctor with this email or registration number already exists"

        password_hash = bcrypt.hashpw(data["password"].encode(), bcrypt.gensalt()).decode()
        doctor = Doctor(
            full_name=data["full_name"],
            hospital_name=data["hospital_name"],
            registration_number=data["registration_number"],
            mobile_number=data["mobile_number"],
            email=data["email"],
            qualification=data.get("qualification", ""),
            specialization=data.get("specialization", ""),
            password_hash=password_hash,
        )
        self.session.add(doctor)
        self.session.commit()
        return doctor, None

    def authenticate(self, email: str, password: str) -> tuple:
        doctor = self.session.query(Doctor).filter_by(email=email).first()
        if not doctor:
            return None, "Invalid email or password"
        if not bcrypt.checkpw(password.encode(), doctor.password_hash.encode()):
            return None, "Invalid email or password"
        if not doctor.is_active:
            return None, "Account is deactivated"
        token = self._generate_token(doctor.id)
        return token, None

    def _generate_token(self, doctor_id: str) -> str:
        payload = {
            "doctor_id": doctor_id,
            "exp": datetime.now(timezone.utc) + timedelta(hours=JWT_EXPIRATION_HOURS),
            "iat": datetime.now(timezone.utc),
        }
        return jwt.encode(payload, SECRET_KEY, algorithm=JWT_ALGORITHM)

    def verify_token(self, token: str) -> tuple:
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[JWT_ALGORITHM])
            doctor = self.session.query(Doctor).get(payload["doctor_id"])
            if not doctor or not doctor.is_active:
                return None, "Doctor not found or inactive"
            return doctor, None
        except jwt.ExpiredSignatureError:
            return None, "Token expired"
        except jwt.InvalidTokenError:
            return None, "Invalid token"

    def change_password(self, doctor_id: str, old_password: str, new_password: str) -> tuple:
        doctor = self.session.query(Doctor).get(doctor_id)
        if not doctor:
            return False, "Doctor not found"
        if not bcrypt.checkpw(old_password.encode(), doctor.password_hash.encode()):
            return False, "Current password is incorrect"
        doctor.password_hash = bcrypt.hashpw(new_password.encode(), bcrypt.gensalt()).decode()
        self.session.commit()
        return True, "Password changed successfully"
