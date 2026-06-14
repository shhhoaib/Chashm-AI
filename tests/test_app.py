"""Quick test script for Chashm AI components."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

def test_database():
    from app.database.connection import DatabaseManager, get_session
    from app.database.models import Doctor
    from app.database.seed import seed_default_doctor

    db = DatabaseManager()
    db.create_tables()
    seed_default_doctor()
    session = get_session()
    count = session.query(Doctor).count()
    doctor = session.query(Doctor).filter_by(email="admin@chashm.ai").first()
    session.close()
    print(f"Database OK: {count} doctors, default: {doctor.full_name}")
    return True

def test_ai_engine():
    from app.ai.engine import ai_engine
    import cv2
    import numpy as np

    img = np.random.randint(100, 200, (300, 300, 3), dtype=np.uint8)
    cv2.imwrite("test_scan.png", img)

    quality = ai_engine.assess_quality("test_scan.png")
    print(f"Quality: score={quality['score']}, passed={quality['passed']}")

    results = ai_engine.analyze("test_scan.png")
    print(f"Analysis: disease_found={results['disease_found']}")
    if results.get("disease"):
        d = results["disease"]
        print(f"  Disease: {d['disease']} ({d['confidence']}%)")
        print(f"  Severity: {results['severity_level']}, Risk: {results['risk_level']}")
        print(f"  Affected: {results['affected_areas']}")
        print(f"  Findings: {results['ai_findings'][:60]}...")

    viz = ai_engine.generate_visualizations("test_scan.png", results, "test_viz")
    for k, v in viz.items():
        print(f"  Viz {k}: {os.path.exists(v)}")

    os.remove("test_scan.png")
    return True

def test_pdf_generator():
    from app.reporting.pdf_generator import generate_report, generate_qr_code

    qr_path = generate_qr_code("TEST-RPT-001")
    print(f"QR Code: {os.path.exists(qr_path)}")

    data = {
        "report_number": "TEST-RPT-001",
        "doctor": {
            "full_name": "Dr. Test",
            "hospital_name": "Test Hospital",
            "registration_number": "REG-001",
        },
        "patient": {
            "patient_id": "P001",
            "name": "John Doe",
            "age": 45,
            "gender": "Male",
            "mobile_number": "+1234567890",
        },
        "scan": {
            "image_path": None,
            "disease": {"disease": "Diabetic Retinopathy", "confidence": 95.5},
            "confidence_score": 95.5,
            "severity_level": "Moderate",
            "risk_level": "Medium",
            "affected_areas": "Retina, Macula",
            "ai_findings": "AI suggests Moderate Diabetic Retinopathy with 95.5% confidence.",
            "quality": {"score": 85.0},
            "ensemble_scores": {
                "Diabetic Retinopathy": {"yolo": 94.0, "vit": 96.0, "efficientnet": 95.0, "convnext": 97.0}
            },
            "heatmap_path": None,
            "grad_cam_path": None,
            "annotated_path": None,
            "segmentation_path": None,
        },
        "qr_code_path": qr_path,
        "follow_up": "Follow-up in 3 months recommended.",
        "clinical_summary": "Moderate DR detected.",
    }
    pdf_path = generate_report(data)
    print(f"PDF Report: {os.path.exists(pdf_path)} at {pdf_path}")
    return True

if __name__ == "__main__":
    print("=" * 50)
    print("Chashm AI - Component Tests")
    print("=" * 50)
    tests = [
        ("Database", test_database),
        ("AI Engine", test_ai_engine),
        ("PDF Generator", test_pdf_generator),
    ]
    all_passed = True
    for name, fn in tests:
        try:
            print(f"\nTesting {name}...")
            fn()
            print(f"  PASS: {name}")
        except Exception as e:
            print(f"  FAIL: {name}: {e}")
            all_passed = False
    print(f"\n{'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
