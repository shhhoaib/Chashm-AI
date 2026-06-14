"""Test AI engine with real PyTorch model loading."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.ai.engine import ai_engine

print("Loading AI models...")
result = ai_engine.load_models()
print(f"Models loaded: {result['loaded']}")
print(f"Count: {result['models_loaded']}/{result['models_attempted']}")
print(f"Device: {result['device']}")
if result["errors"]:
    print(f"Errors: {result['errors']}")
print(f"Models available: {list(ai_engine.models.keys())}")

# Test analysis with real models
import cv2
import numpy as np
from PIL import Image

img = np.random.randint(100, 200, (300, 300, 3), dtype=np.uint8)
Image.fromarray(img).save("test_model.png")

quality = ai_engine.assess_quality("test_model.png")
print(f"Quality: {quality['score']}% passed={quality['passed']}")

results = ai_engine.analyze("test_model.png")
print(f"Disease found: {results['disease_found']}")
print(f"Models used: {results.get('models_used')}")
if results.get("disease"):
    d = results["disease"]
    print(f"Primary: {d['disease']} ({d['confidence']}%)")

# Test visualizations with Grad-CAM
print("Generating visualizations...")
viz = ai_engine.generate_visualizations("test_model.png", results, "test_viz2")
for k, v in viz.items():
    print(f"  Viz {k}: {os.path.exists(v)}")

os.remove("test_model.png")
print("Real model test passed!")
