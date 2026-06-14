from app.utils.helpers import validate_email, validate_phone


def validate_doctor_registration(data: dict) -> tuple:
    errors = []
    if not data.get("full_name"):
        errors.append("Full name is required")
    if not data.get("hospital_name"):
        errors.append("Hospital name is required")
    if not data.get("registration_number"):
        errors.append("Registration number is required")
    if not data.get("mobile_number") or not validate_phone(data["mobile_number"]):
        errors.append("Valid mobile number is required")
    if not data.get("email") or not validate_email(data["email"]):
        errors.append("Valid email is required")
    if not data.get("password") or len(data["password"]) < 6:
        errors.append("Password must be at least 6 characters")
    return (True, None) if not errors else (False, errors)


def validate_patient_data(data: dict) -> tuple:
    errors = []
    if not data.get("name"):
        errors.append("Patient name is required")
    if not data.get("age") or not isinstance(data.get("age"), int):
        errors.append("Valid age is required")
    if data.get("age") and (data["age"] < 0 or data["age"] > 150):
        errors.append("Age must be between 0 and 150")
    return (True, None) if not errors else (False, errors)
