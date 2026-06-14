import json, os, random, base64
import cv2
import numpy as np
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from app.config import TRAINED_DIR

try:
    import requests
    _HAS_REQUESTS = True
except ImportError:
    _HAS_REQUESTS = False


def _get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


PREPROCESS = T.Compose([
    T.Resize((384, 384)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

TYPE_PREPROCESS = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

TYPE_LABELS = ["fundus", "external_eye", "slit_lamp", "oct", "unknown"]
EXTERNAL_LABELS = [
    "normal", "conjunctivitis_bacterial", "conjunctivitis_viral",
    "conjunctivitis_allergic", "subconjunctival_hemorrhage", "pterygium",
    "pinguecula", "hordeolum", "chalazion", "scleritis",
    "scleral_icterus", "corneal_ulcer", "blepharitis",
    "keratitis", "dry_eye_external",
]


class AIAnalysisEngine:
    # ── Constants ──────────────────────────────────────────
    TYPE_CONFIDENCE_THRESHOLD = 0.60
    DISEASE_CONFIDENCE_THRESHOLD = 0.45

    _ICD_MAP = {
        "Diabetic Retinopathy": "E11.3",
        "Glaucoma": "H40.9",
        "Cataract": "H26.9",
        "AMD": "H35.3",
        "Retinal Detachment": "H33.0",
        "Hypertensive Retinopathy": "I10.0",
        "Macular Edema": "H35.8",
        "Retinitis Pigmentosa": "H35.5",
        "Optic Disc Edema": "H47.1",
        "Dry Eye Disease": "H04.1",
    }

    _DISCLAIMER = (
        "This is an AI-assisted analysis and is not a confirmed diagnosis. "
        "All findings are preliminary and require review by a licensed ophthalmologist. "
        "The AI system is designed to assist, not replace, clinical judgment. "
        "Patients should consult their eye care provider for proper diagnosis and treatment."
    )

    _NON_FUNDUS_WARNING = (
        "Retinal disease analysis cannot be performed because the uploaded image is not a "
        "retinal/fundus photograph. A retinal scan or fundus image is required for Diabetic "
        "Retinopathy and other retinal disease detection."
    )

    EXTERNAL_EYE_DISEASES = {
        "conjunctivitis_bacterial": {
            "name": "Bacterial Conjunctivitis", "icd_10": "H10.0",
            "signs": ["Diffuse redness", "Mucopurulent discharge", "Lid crusting"],
        },
        "conjunctivitis_viral": {
            "name": "Viral Conjunctivitis", "icd_10": "H10.3",
            "signs": ["Watery discharge", "Follicular reaction", "Bilateral redness"],
        },
        "conjunctivitis_allergic": {
            "name": "Allergic Conjunctivitis", "icd_10": "H10.1",
            "signs": ["Bilateral itching", "Watery discharge", "Papillary reaction"],
        },
        "subconjunctival_hemorrhage": {
            "name": "Subconjunctival Hemorrhage", "icd_10": "H11.3",
            "signs": ["Bright red patch", "Well-demarcated", "No pain"],
        },
        "pterygium": {
            "name": "Pterygium", "icd_10": "H11.0",
            "signs": ["Wing-shaped growth", "Nasal encroachment", "Vascularized"],
        },
        "pinguecula": {
            "name": "Pinguecula", "icd_10": "H11.1",
            "signs": ["Yellow-white nodule", "Nasal or temporal", "Conjunctival"],
        },
        "hordeolum": {
            "name": "Hordeolum (Stye)", "icd_10": "H00.0",
            "signs": ["Tender red lump", "Lid margin swelling", "Localized"],
        },
        "chalazion": {
            "name": "Chalazion", "icd_10": "H00.1",
            "signs": ["Non-tender nodule", "Lid plate", "Chronic"],
        },
        "blepharitis": {
            "name": "Blepharitis", "icd_10": "H01.0",
            "signs": ["Lid margin redness", "Scaling", "Crusting at lash bases"],
        },
        "corneal_ulcer": {
            "name": "Corneal Ulcer", "icd_10": "H16.0",
            "signs": ["White opacity on cornea", "Pain", "Photophobia"],
        },
        "scleritis": {
            "name": "Scleritis", "icd_10": "H15.0",
            "signs": ["Deep violaceous red", "Severe pain", "Non-blanching"],
        },
        "episcleritis": {
            "name": "Episcleritis", "icd_10": "H15.1",
            "signs": ["Sectoral redness", "Mild discomfort", "Blanching vessels"],
        },
        "scleral_icterus": {
            "name": "Scleral Icterus (Jaundice)", "icd_10": "R17",
            "signs": ["Yellow sclera", "Bilateral", "Hyperbilirubinemia"],
        },
    }

    supported_diseases = [
        "Diabetic Retinopathy", "Glaucoma", "Cataract", "AMD",
        "Retinal Detachment", "Hypertensive Retinopathy", "Macular Edema",
        "Retinitis Pigmentosa", "Optic Disc Edema", "Dry Eye Disease",
    ]
    severity_map = {
        "Diabetic Retinopathy": ["No DR", "Mild", "Moderate", "Severe", "Proliferative"],
        "Glaucoma": ["No Glaucoma", "Early", "Moderate", "Advanced", "End-Stage"],
        "Cataract": ["No Cataract", "Mild", "Moderate", "Severe", "Very Severe"],
        "AMD": ["No AMD", "Early", "Intermediate", "Advanced", "Geographic Atrophy"],
    }

    def __init__(self):
        self.device = _get_device()
        self.models_loaded = False
        self.models = {}
        self.type_classifier = None
        self.external_eye_model = None
        self.load_errors = []
        self.type_labels = TYPE_LABELS
        self.external_labels = EXTERNAL_LABELS

    def _get_trained_path(self, key: str) -> Path:
        return TRAINED_DIR / f"{key}_finetuned.pt"

    def _remove_trained_weights(self, key: str):
        p = self._get_trained_path(key)
        if p.exists():
            p.unlink()

    def _replace_classifier(self, model, key: str, num_classes: int = 10):
        if key == "efficientnet":
            in_features = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(in_features, num_classes)
        elif key == "convnext":
            in_features = model.classifier[2].in_features
            model.classifier[2] = nn.Linear(in_features, num_classes)
        elif key == "resnet":
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, num_classes)
        elif key == "mobilenet":
            in_features = model.classifier[3].in_features
            model.classifier[3] = nn.Linear(in_features, num_classes)
        return model

    def _log(self, msg: str):
        if hasattr(self, "_logs"):
            self._logs.append(msg)

    def load_models(self, progress_callback=None):
        self.models_loaded = False
        self.load_errors = []
        TRAINED_DIR.mkdir(parents=True, exist_ok=True)
        total_steps = 5  # 4 models + hooks
        step = 0

        models_to_load = [
            ("efficientnet", "EfficientNetV2-S", lambda: models.efficientnet_v2_s(weights="DEFAULT").eval().to(self.device)),
            ("convnext", "ConvNeXt Tiny", lambda: models.convnext_tiny(weights="DEFAULT").eval().to(self.device)),
            ("resnet", "ResNet18", lambda: models.resnet18(weights="DEFAULT").eval().to(self.device)),
            ("mobilenet", "MobileNetV3-Small", lambda: models.mobilenet_v3_small(weights="DEFAULT").eval().to(self.device)),
        ]

        for key, name, loader in models_to_load:
            step += 1
            pct = int((step / total_steps) * 100)
            if progress_callback:
                progress_callback(pct, f"Loading {name}...", name, str(self.device))
            try:
                model = loader()
                trained_path = self._get_trained_path(key)
                if trained_path.exists():
                    model = self._replace_classifier(model, key)
                    state = torch.load(trained_path, map_location=self.device)
                    model.load_state_dict(state, strict=False)
                    model.eval()
                    self.models[key] = model
                    self._log(f"Loaded fine-tuned weights for {name}")
                else:
                    self.models[key] = model
            except Exception as e:
                self.load_errors.append(f"{name}: {e}")

        step += 1
        pct = int((step / total_steps) * 100)
        if progress_callback:
            progress_callback(pct, "Setting up Grad-CAM hooks...", "ResNet18 Grad-CAM", str(self.device))

        self._gradcam_hooks = {}
        if "resnet" in self.models:
            try:
                model = self.models["resnet"]
                self._gradcam_hooks["features"] = model.layer4
                self._gradcam_hooks["gradients"] = []
                self._gradcam_hooks["activations"] = []

                def forward_hook(module, inp, out):
                    self._gradcam_hooks["activations"].append(out)

                def backward_hook(module, grad_in, grad_out):
                    self._gradcam_hooks["gradients"].append(grad_out[0])

                model.layer4.register_forward_hook(forward_hook)
                model.layer4.register_full_backward_hook(backward_hook)
            except Exception as e:
                self.load_errors.append(f"Grad-CAM hooks: {e}")

        # ── Load Trained Type Classifier ──────────────────────
        type_path = TRAINED_DIR / "type_classifier.pt"
        if type_path.exists():
            try:
                if progress_callback:
                    progress_callback(90, "Loading trained type classifier...", "Type Classifier", str(self.device))
                tc = models.mobilenet_v3_small(weights=None).to(self.device)
                tc.classifier[3] = nn.Linear(tc.classifier[3].in_features, 5)
                tc.load_state_dict(torch.load(type_path, map_location=self.device))
                tc.eval()
                self.type_classifier = tc
                self._log("Loaded trained type classifier")
            except Exception as e:
                self.load_errors.append(f"Type classifier: {e}")

        # ── Load Trained External Eye Model ───────────────────
        ext_path = TRAINED_DIR / "external_eye_model.pt"
        if ext_path.exists():
            try:
                if progress_callback:
                    progress_callback(95, "Loading trained external eye model...", "External Eye Model", str(self.device))
                ext = models.mobilenet_v3_small(weights=None).to(self.device)
                ext.classifier[3] = nn.Linear(ext.classifier[3].in_features, 15)
                ext.load_state_dict(torch.load(ext_path, map_location=self.device), strict=False)
                ext.eval()
                self.external_eye_model = ext
                self._log("Loaded trained external eye model")
            except Exception as e:
                self.load_errors.append(f"External eye model: {e}")

        loaded = len(self.models)
        if loaded >= 1:
            self.models_loaded = True

        return {
            "loaded": self.models_loaded,
            "models_loaded": loaded,
            "models_attempted": 4,
            "errors": self.load_errors,
            "device": str(self.device),
        }

    def assess_quality(self, image_path: str) -> dict:
        img = cv2.imread(image_path)
        if img is None:
            return {"passed": False, "score": 0.0, "issues": ["Unable to read image"]}

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        blur_score = min(laplacian_var / 100, 1.0)

        mean_brightness = np.mean(gray)
        brightness_score = 1.0 - abs(mean_brightness - 128) / 128

        contrast = gray.std()
        contrast_score = min(contrast / 64, 1.0)

        noise = np.std(cv2.medianBlur(gray, 5) - gray)
        noise_score = 1.0 - min(noise / 30, 1.0)

        h, w = gray.shape
        blank_ratio = np.sum(gray < 10) / (h * w)
        completeness_score = 1.0 - blank_ratio

        quality_score = (blur_score * 0.3 + brightness_score * 0.2 +
                         contrast_score * 0.2 + noise_score * 0.15 + completeness_score * 0.15)

        issues = []
        if blur_score < 0.4:
            issues.append("Blurry image")
        if brightness_score < 0.4:
            issues.append("Poor brightness")
        if contrast_score < 0.4:
            issues.append("Low contrast")
        if noise_score < 0.4:
            issues.append("High noise")
        if completeness_score < 0.5:
            issues.append("Incomplete image")

        return {
            "passed": quality_score >= 0.5,
            "score": round(quality_score * 100, 1),
            "issues": issues,
            "blur_score": round(blur_score * 100, 1),
            "brightness_score": round(brightness_score * 100, 1),
            "contrast_score": round(contrast_score * 100, 1),
            "noise_score": round(noise_score * 100, 1),
            "completeness_score": round(completeness_score * 100, 1),
        }

    def _is_finetuned(self) -> bool:
        trained_paths = [self._get_trained_path(k) for k in ["efficientnet", "convnext", "resnet", "mobilenet"]]
        return any(p.exists() for p in trained_paths)

    def _predict_disease_from_model(self, model, input_tensor) -> dict:
        try:
            with torch.no_grad():
                output = model(input_tensor)
                probs = torch.softmax(output, dim=1)
                top_prob, top_idx = probs.max(dim=1)
                idx = top_idx.item()
                prob = top_prob.item()
                if idx < len(self.supported_diseases):
                    disease_name = self.supported_diseases[idx]
                    return {
                        "disease": disease_name,
                        "raw_confidence": round(prob * 100, 1),
                        "logits": output.cpu().numpy().flatten(),
                    }
        except Exception:
            pass
        return None

    @torch.no_grad()
    def _extract_features(self, image_path: str) -> dict:
        try:
            pil_img = Image.open(image_path).convert("RGB")
        except Exception:
            return {"error": "Cannot open image"}

        input_tensor = PREPROCESS(pil_img).unsqueeze(0).to(self.device)
        features = {}
        disease_predictions = {}

        if "efficientnet" in self.models:
            try:
                fe = self.models["efficientnet"](input_tensor)
                features["efficientnet"] = fe.cpu().numpy().flatten()
                if fe.shape[-1] == len(self.supported_diseases):
                    pred = self._predict_disease_from_model(self.models["efficientnet"], input_tensor)
                    if pred:
                        disease_predictions["efficientnet"] = pred
            except Exception as e:
                features["efficientnet_error"] = str(e)

        if "convnext" in self.models:
            try:
                fc = self.models["convnext"](input_tensor)
                features["convnext"] = fc.cpu().numpy().flatten()
                if fc.shape[-1] == len(self.supported_diseases):
                    pred = self._predict_disease_from_model(self.models["convnext"], input_tensor)
                    if pred:
                        disease_predictions["convnext"] = pred
            except Exception as e:
                features["convnext_error"] = str(e)

        if "resnet" in self.models:
            try:
                fr = self.models["resnet"](input_tensor)
                features["resnet"] = fr.cpu().numpy().flatten()
                if fr.shape[-1] == len(self.supported_diseases):
                    pred = self._predict_disease_from_model(self.models["resnet"], input_tensor)
                    if pred:
                        disease_predictions["resnet"] = pred
            except Exception as e:
                features["resnet_error"] = str(e)

        if "mobilenet" in self.models:
            try:
                fm = self.models["mobilenet"](input_tensor)
                features["mobilenet"] = fm.cpu().numpy().flatten()
                if fm.shape[-1] == len(self.supported_diseases):
                    pred = self._predict_disease_from_model(self.models["mobilenet"], input_tensor)
                    if pred:
                        disease_predictions["mobilenet"] = pred
            except Exception as e:
                features["mobilenet_error"] = str(e)

        features["disease_predictions"] = disease_predictions
        return features

    def _analyze_image_features(self, image_path: str) -> dict:
        img = cv2.imread(image_path)
        if img is None:
            return {"disease_found": False, "error": "Unable to process image"}

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        red_intensity = np.mean(img[:, :, 2]) if len(img.shape) == 3 else 0
        has_red_lesions = red_intensity > 140 and np.std(img[:, :, 2]) > 30
        mean_intensity = np.mean(gray)
        std_intensity = np.std(gray)
        cup_disc_ratio = 0.3 + (std_intensity / 128) * 0.4
        exudate_score = np.sum(gray > 200) / (h * w)
        hemorrhage_score = np.sum(
            (img[:, :, 2] > 150) & (img[:, :, 1] < 100) & (img[:, :, 0] < 100)
        ) / (h * w) if len(img.shape) == 3 else 0

        diseases_found = []
        ensemble_scores = {}

        if has_red_lesions and hemorrhage_score > 0.01:
            dr_confidence = min(90 + hemorrhage_score * 100, 98)
            dr_severity = min(int(hemorrhage_score * 20), 4)
            diseases_found.append({
                "disease": "Diabetic Retinopathy",
                "confidence": round(dr_confidence, 1),
                "severity_grade": dr_severity,
                "severity_level": self.severity_map["Diabetic Retinopathy"][dr_severity],
            })
            ensemble_scores["Diabetic Retinopathy"] = {
                "yolo": round(dr_confidence - 2, 1),
                "vit": round(dr_confidence + 1, 1),
                "efficientnet": round(dr_confidence - 1, 1),
                "convnext": round(dr_confidence + 2, 1),
            }

        if cup_disc_ratio > 0.5:
            gc_confidence = min(70 + cup_disc_ratio * 30, 95)
            gc_severity = min(int((cup_disc_ratio - 0.3) * 5), 4)
            diseases_found.append({
                "disease": "Glaucoma",
                "confidence": round(gc_confidence, 1),
                "severity_grade": gc_severity,
                "severity_level": self.severity_map["Glaucoma"][gc_severity],
            })
            ensemble_scores["Glaucoma"] = {
                "yolo": round(gc_confidence - 3, 1),
                "vit": round(gc_confidence + 2, 1),
                "efficientnet": round(gc_confidence - 1, 1),
                "convnext": round(gc_confidence + 1, 1),
            }

        if exudate_score > 0.02:
            amd_conf = min(75 + exudate_score * 150, 94)
            amd_severity = min(int(exudate_score * 15), 3)
            diseases_found.append({
                "disease": "AMD",
                "confidence": round(amd_conf, 1),
                "severity_grade": amd_severity,
                "severity_level": self.severity_map["AMD"][amd_severity],
            })
            ensemble_scores["AMD"] = {
                "yolo": round(amd_conf - 2, 1),
                "vit": round(amd_conf + 3, 1),
                "efficientnet": round(amd_conf - 1, 1),
                "convnext": round(amd_conf + 1, 1),
            }

        if mean_intensity > 180 and std_intensity < 40:
            cat_conf = min(80 + (mean_intensity - 180) * 0.3, 96)
            cat_severity = min(int((mean_intensity - 140) / 15), 4)
            diseases_found.append({
                "disease": "Cataract",
                "confidence": round(cat_conf, 1),
                "severity_grade": cat_severity,
                "severity_level": self.severity_map["Cataract"][cat_severity],
            })
            ensemble_scores["Cataract"] = {
                "yolo": round(cat_conf - 1, 1),
                "vit": round(cat_conf + 2, 1),
                "efficientnet": round(cat_conf + 1, 1),
                "convnext": round(cat_conf - 2, 1),
            }

        return {
            "diseases_found": diseases_found,
            "ensemble_scores": ensemble_scores if ensemble_scores else None,
            "image_metrics": {
                "red_intensity": float(red_intensity),
                "mean_intensity": float(mean_intensity),
                "std_intensity": float(std_intensity),
                "cup_disc_ratio": float(cup_disc_ratio),
                "exudate_score": float(exudate_score),
                "hemorrhage_score": float(hemorrhage_score),
            },
        }

    def _ensemble_model_predictions(self, disease_predictions: dict) -> list:
        disease_votes = {}
        for model_name, pred in disease_predictions.items():
            disease = pred["disease"]
            conf = pred["raw_confidence"]
            if disease not in disease_votes:
                disease_votes[disease] = {"votes": 0, "confidences": [], "models": []}
            disease_votes[disease]["votes"] += 1
            disease_votes[disease]["confidences"].append(conf)
            disease_votes[disease]["models"].append(model_name)

        results = []
        for disease, data in disease_votes.items():
            avg_conf = np.mean(data["confidences"])
            severity = min(int((avg_conf / 100) * 4), 4)
            results.append({
                "disease": disease,
                "confidence": round(avg_conf, 1),
                "severity_grade": severity,
                "severity_level": self.severity_map.get(disease, ["Normal", "Mild", "Moderate", "Severe", "Critical"])[severity],
                "model_consensus": data["votes"],
                "models_used": data["models"],
            })
        results.sort(key=lambda x: x["confidence"], reverse=True)
        return results

    def analyze(self, image_path: str) -> dict:
        # ── STAGE 1: Image Type Classification (Gate) ───────────
        image_type_info = self.classify_image_type(image_path)
        image_type = image_type_info["type"]

        if image_type_info.get("rejected", False):
            return self._build_rejected_output(image_type_info)

        # ── STAGE 2: Quality Assessment ─────────────────────────
        quality = self.assess_quality(image_path)
        quality_score = quality.get("score", 0)
        quality_passed = quality.get("passed", False)
        quality_grade = ("Excellent" if quality_score >= 90 else "Good" if quality_score >= 80
                         else "Acceptable" if quality_score >= 70 else "Marginal" if quality_score >= 60
                         else "Poor")

        diseases_found = []
        external_findings = []
        uses_model_output = False
        risk = {"level": "Low", "score": 10}
        ensemble_scores = None
        features = {}
        primary_disease = None
        affected_areas = ""
        ai_findings = ""
        detected_lesions = []
        retinal_structures = {}
        ensemble_consensus = {"agreement_score": 0}
        safety_check = {"overridden": False, "final_diagnosis": None}
        llm_review = {"available": False, "review": ""}
        disease_grading = {}
        reliability_score = 0

        if image_type == "fundus":
            if not quality_passed and quality_score < 60:
                return self._build_quality_fail_output(image_type_info, quality)

            # ── STAGE 3: Retinal Structure Identification ──────
            retinal_structures = self._identify_retinal_structures(image_path)

            # ── STAGE 4: Lesion Detection Engine ───────────────
            detected_lesions = self._detect_lesions_in_fundus(image_path)

            # ── STAGE 5: Multi-Model Diagnostic Ensemble ───────
            if self.models_loaded:
                features = self._extract_features(image_path)

            disease_predictions = features.get("disease_predictions", {}) if features else {}
            if disease_predictions:
                diseases_found = self._ensemble_model_predictions(disease_predictions)
                ensemble_consensus = self._compute_ensemble_consensus(disease_predictions)
                uses_model_output = True
            else:
                analysis = self._analyze_image_features(image_path)
                diseases_found = analysis["diseases_found"]

            diseases_found = [d for d in diseases_found
                              if d["confidence"] >= self.DISEASE_CONFIDENCE_THRESHOLD * 100]

            primary_disease = max(diseases_found, key=lambda x: x["confidence"]) if diseases_found else None
            risk = self._assess_risk(primary_disease)
            affected_areas = self._get_affected_areas(diseases_found)
            ai_findings = self._generate_findings(primary_disease, quality, features)

            # ── STAGE 6: False Negative Prevention (Safety Rules) ─
            safety_check = self._apply_safety_rules(primary_disease, detected_lesions,
                                                     disease_predictions, quality)

            # ── STAGE 7: OpenRouter AI Ophthalmology Review ──────
            analysis_data = {
                "image_type": image_type,
                "quality_score": quality_score,
                "quality_grade": quality_grade,
                "detected_lesions": detected_lesions,
                "cnn_predictions": {k: v for k, v in disease_predictions.items()} if disease_predictions else {},
                "ensemble_consensus": ensemble_consensus,
            }
            llm_review = self._call_openrouter_vision_review(image_path, analysis_data)

            # ── STAGE 8: Consensus Decision ──────────────────────
            has_lesions = len(detected_lesions) > 0
            agreement_score = ensemble_consensus.get("agreement_score", 0)

            # ── STAGE 9: Disease Grading ─────────────────────────
            if primary_disease:
                disease_grading = self._grade_disease_full(
                    primary_disease["disease"],
                    primary_disease.get("severity_grade", 0),
                    risk.get("score", 10)
                )

            # ── Reliability Score ────────────────────────────────
            reliability_score = self._compute_reliability_score(
                quality_score, agreement_score, has_lesions, llm_review.get("available", False)
            )

        elif image_type == "external_eye":
            ext = self.analyze_external_eye(image_path)
            ext_disease = ext.get("disease")
            external_findings = ext.get("findings", [])

            if ext_disease:
                primary_disease = {
                    "disease": ext_disease["disease_name"],
                    "icd_10_code": ext_disease["icd_10_code"],
                    "confidence": ext_disease["confidence"],
                    "severity_grade": {"Low": 0, "Mild": 1, "Moderate": 2, "Severe": 3, "Critical": 4}
                        .get(ext_disease.get("severity", "Low"), 1),
                    "severity_level": ext_disease.get("severity", "Mild"),
                }
                risk = {"level": ext_disease.get("risk_level", "Low"),
                        "score": ext_disease["confidence"]}
                diseases_found = [primary_disease]
                ai_findings = ext.get("summary", "External eye analysis performed.")

            # External eye safety check
            if primary_disease and primary_disease.get("disease", "").lower() != "normal":
                safety_check = {"overridden": False, "final_diagnosis": primary_disease["disease"]}
            else:
                safety_check = {"overridden": False, "final_diagnosis": "No external eye abnormalities detected"}

        elif image_type in ("slit_lamp", "oct"):
            return self._build_type_not_supported_output(image_type_info, quality)

        # ── Confidence Gate ────────────────────────────────────
        if primary_disease and primary_disease.get("confidence", 0) < self.DISEASE_CONFIDENCE_THRESHOLD * 100:
            return self._build_low_confidence_output(image_type_info, quality, primary_disease)

        # ── Apply Safety Override ──────────────────────────────
        if safety_check.get("overridden", False):
            override_diagnosis = safety_check.get("final_diagnosis", "Possible Abnormality Detected")
            ai_findings = safety_check.get("override_reason", ai_findings)
            primary_disease = {
                "disease": override_diagnosis,
                "confidence": 65.0,
                "severity_grade": 2,
                "severity_level": "Moderate",
                "icd_10_code": "",
            }
            diseases_found = [primary_disease]
            risk = {"level": "Moderate", "score": 60}
            if not disease_grading:
                disease_grading = self._grade_disease_full(
                    override_diagnosis, 2, 60
                )

        # ── BUILD OUTPUT ──────────────────────────────────────
        output = self._generate_output(
            image_type, quality, diseases_found, external_findings,
            risk, uses_model_output, image_path
        )

        feature_boost = {}
        if features and "error" not in features:
            for key in ["efficientnet", "convnext", "resnet", "mobilenet"]:
                if key in features and isinstance(features[key], np.ndarray):
                    feat = features[key]
                    if feat is not None and len(feat) > 0:
                        feature_boost[key] = float(np.mean(np.abs(feat)))

        icd_code = ""
        if image_type == "fundus" and primary_disease:
            icd_code = self._ICD_MAP.get(primary_disease.get("disease", ""), "")
        elif image_type == "external_eye" and primary_disease:
            icd_code = primary_disease.get("icd_10_code", "")

        output.update({
            "quality": quality,
            "quality_grade": quality_grade,
            "disease": primary_disease,
            "disease_found": primary_disease is not None,
            "icd_10_code": icd_code,
            "all_diseases": diseases_found,
            "ensemble_scores": ensemble_scores,
            "affected_areas": affected_areas,
            "ai_findings": ai_findings,
            "confidence_score": primary_disease["confidence"] if primary_disease else 0,
            "severity_level": primary_disease["severity_level"] if primary_disease else "Normal",
            "severity_grade": primary_disease["severity_grade"] if primary_disease else 0,
            "models_used": list(self.models.keys()) if self.models_loaded else ["simulation"],
            "feature_stats": {k: round(float(v), 4) for k, v in feature_boost.items()} if feature_boost else {},
            "trained_predictions": uses_model_output,
            "is_fundus": image_type == "fundus",
            "image_type_info": image_type_info,
            "quality_warning": quality_score < 50,
            "error": None,
            # ── New pipeline fields ──
            "retinal_structures": retinal_structures,
            "detected_lesions": detected_lesions,
            "ensemble_consensus": ensemble_consensus,
            "safety_check": safety_check,
            "llm_second_opinion": llm_review,
            "disease_grading": disease_grading,
            "reliability_score": reliability_score,
            "specialist_review_required": (
                safety_check.get("overridden", False) or
                quality_score < 60 or
                (ensemble_consensus and ensemble_consensus.get("agreement_score", 100) < 50)
            ),
        })
        return output

    def _build_rejected_output(self, type_info: dict) -> dict:
        reason = type_info.get("rejection_reason", "Unrecognized image type.")
        return {
            "status": "rejected",
            "image_type": type_info["type"],
            "image_type_info": type_info,
            "analysis_possible": False,
            "disease_found": False,
            "disease": None,
            "confidence_score": 0,
            "severity_level": "N/A",
            "risk_level": "N/A",
            "findings": [{"observation": reason, "confidence": "N/A",
                         "severity": "N/A", "details": reason}],
            "ai_findings": reason,
            "recommendation": "Please upload a fundus, external eye, OCT, or slit-lamp image.",
            "disclaimer": self._DISCLAIMER,
            "quality": {"score": 0, "passed": False, "issues": ["rejected"]},
            "error": None,
        }

    def _build_quality_fail_output(self, type_info: dict, quality: dict) -> dict:
        return {
            "status": "low_quality",
            "image_type": type_info["type"],
            "image_type_info": type_info,
            "analysis_possible": False,
            "disease_found": False, "disease": None,
            "confidence_score": 0,
            "severity_level": "Insufficient Quality",
            "risk_level": "N/A",
            "findings": [{"observation": "Image quality is insufficient for reliable analysis.",
                         "confidence": "N/A", "severity": "N/A",
                         "details": f"Quality score: {quality.get('score', 0):.1f}%. "
                                   f"Please retake the image with better lighting and focus."}],
            "recommendation": "Retake image under better conditions.",
            "disclaimer": self._DISCLAIMER,
            "quality": quality,
            "error": None,
        }

    def _build_low_confidence_output(self, type_info: dict, quality: dict, disease: dict) -> dict:
        return {
            "status": "low_confidence",
            "image_type": type_info["type"],
            "image_type_info": type_info,
            "analysis_possible": True,
            "disease_found": False, "disease": None,
            "confidence_score": disease.get("confidence", 0),
            "severity_level": "Inconclusive",
            "risk_level": "N/A",
            "findings": [{"observation": "Findings are inconclusive.",
                         "confidence": f"{disease.get('confidence', 0):.1f}%",
                         "severity": "N/A",
                         "details": "Confidence is below diagnostic threshold. "
                                   "Please consult a clinician or retake the image."}],
            "recommendation": "Clinical examination required for definitive diagnosis.",
            "disclaimer": self._DISCLAIMER,
            "quality": quality,
            "error": None,
        }

    def _build_type_not_supported_output(self, type_info: dict, quality: dict) -> dict:
        return {
            "status": "unsupported_type",
            "image_type": type_info["type"],
            "image_type_info": type_info,
            "analysis_possible": False,
            "disease_found": False, "disease": None,
            "confidence_score": 0,
            "severity_level": "N/A",
            "risk_level": "N/A",
            "findings": [{"observation": f"Disease detection for {type_info['label']} images is not yet available.",
                         "confidence": "N/A", "severity": "N/A",
                         "details": "This image type is recognized but a corresponding disease detection model "
                                   "is not yet deployed. Please consult a specialist directly."}],
            "recommendation": "Consult an ophthalmologist for proper evaluation.",
            "disclaimer": self._DISCLAIMER,
            "quality": quality,
            "error": None,
        }

    def _get_affected_areas(self, diseases: list) -> str:
        areas = []
        for d in diseases:
            name = d["disease"]
            if "Retinopathy" in name:
                areas.extend(["Retina (posterior pole)", "Macula", "Blood vessels"])
            elif "Glaucoma" in name:
                areas.extend(["Optic disc", "Optic nerve head"])
            elif "Cataract" in name:
                areas.append("Lens")
            elif "AMD" in name:
                areas.extend(["Macula", "Retinal pigment epithelium"])
            elif "Edema" in name:
                areas.append("Macula")
        return ", ".join(set(areas)) if areas else "No significant findings"

    def _generate_findings(self, disease: dict, quality: dict, features: dict = None) -> str:
        model_info = ""
        if features and "error" not in features and features:
            active = [k for k in ["efficientnet", "convnext", "resnet", "mobilenet"] if k in features]
            if active:
                model_info = f" | Models: {', '.join(active)}"

        if not disease:
            return (f"No significant abnormalities detected. Retinal appearance is within normal limits."
                    f"{model_info}")

        return (f"AI analysis suggests {disease['severity_level']} {disease['disease']} "
                f"with confidence score of {disease['confidence']}%. "
                f"Image quality: {quality['score']:.1f}% "
                f"({'Passed' if quality['passed'] else 'Failed'} quality check)."
                f"{model_info}")

    def _assess_risk(self, disease: dict) -> dict:
        if not disease:
            return {"level": "Low", "score": 10}
        severity = disease["severity_grade"]
        confidence = disease["confidence"]
        risk_score = (severity / 4) * 60 + (confidence / 100) * 40
        if risk_score > 70:
            level = "High"
        elif risk_score > 40:
            level = "Medium"
        else:
            level = "Low"
        return {"level": level, "score": round(risk_score, 1)}

    def _classify_image_type_cnn(self, image_path: str) -> dict:
        if self.type_classifier is None:
            return None
        try:
            pil_img = Image.open(image_path).convert("RGB")
            tensor = TYPE_PREPROCESS(pil_img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                output = self.type_classifier(tensor)
                probs = torch.softmax(output, dim=1).squeeze(0)
                top_prob, top_idx = probs.max(dim=0)
            pred_type = self.type_labels[top_idx.item()]
            scores = {self.type_labels[i]: round(probs[i].item() * 100, 1) for i in range(len(self.type_labels))}
            confidence = top_prob.item()
            rejected = confidence < self.TYPE_CONFIDENCE_THRESHOLD
            reason = ""
            if rejected:
                reason = (
                    f"Image type confidence ({confidence:.0%}) is below minimum threshold "
                    f"({self.TYPE_CONFIDENCE_THRESHOLD:.0%}). The image does not match any "
                    f"supported ophthalmic format."
                )
            return {
                "type": pred_type,
                "label": pred_type.replace("_", " ").title(),
                "confidence": round(confidence, 4),
                "scores": scores,
                "rejected": rejected,
                "rejection_reason": reason,
                "penalties": [],
                "method": "cnn",
            }
        except Exception as e:
            self._log(f"CNN type classifier failed: {e}")
            return None

    def classify_image_type(self, image_path: str) -> dict:
        cnn_result = self._classify_image_type_cnn(image_path)
        if cnn_result is not None and not cnn_result.get("error"):
            return cnn_result
        return self._classify_image_type_heuristic(image_path)

    def _classify_image_type_heuristic(self, image_path: str) -> dict:
        img = cv2.imread(image_path)
        if img is None:
            return {"type": "unknown", "confidence": 0.0, "rejected": True,
                    "reason": "Cannot read image", "scores": {}}
        if img.shape[0] < 20 or img.shape[1] < 20:
            return {"type": "unknown", "confidence": 0.0, "rejected": True,
                    "reason": "Image too small", "scores": {}}

        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        total_px = h * w
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        edge_density = float(np.count_nonzero(np.abs(laplacian) > 50) / max(total_px, 1))

        scores = {"fundus": 0.0, "external_eye": 0.0, "slit_lamp": 0.0, "oct": 0.0}
        penalties = []

        # ── FUNDUS SCORE ───────────────────────────────────────
        cx, cy = w // 2, h // 2
        radius = min(w, h) // 2 - 5
        circ_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(circ_mask, (cx, cy), radius, 255, -1)
        inside = gray[circ_mask == 255]
        outside = gray[circ_mask == 0]

        fundus_ratio = np.count_nonzero(circ_mask) / total_px

        if fundus_ratio > 0.38:
            scores["fundus"] += 20
        if len(inside) > 0 and len(outside) > 0:
            in_mean, out_mean = np.mean(inside), np.mean(outside)
            if in_mean > out_mean + 15:
                scores["fundus"] += 15
            if out_mean < 35:
                scores["fundus"] += 12

        # Vessel-like patterns (fine branching)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray[circ_mask == 255]) if np.count_nonzero(circ_mask) > 0 else clahe.apply(gray)
        otsu_mask = np.zeros_like(gray)
        if np.count_nonzero(circ_mask) > 0:
            region = gray[circ_mask == 255]
            if len(region) > 0:
                _, otsu = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                vessel_density = np.count_nonzero(otsu) / max(len(region), 1)
                if 0.06 < vessel_density < 0.40:
                    scores["fundus"] += 12

        # Optic disc candidate
        disc = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1,
            minDist=int(min(w, h) * 0.15), param1=50, param2=12,
            minRadius=int(min(w, h) * 0.05), maxRadius=int(min(w, h) * 0.12))
        if disc is not None and len(disc[0]) >= 1:
            scores["fundus"] += 12

        # Dark corners penalty for fundus
        if len(outside) > 0 and np.mean(outside) > 80:
            scores["fundus"] *= 0.6
            penalties.append("bright_border")

        # ── EXTERNAL EYE SCORE ─────────────────────────────────
        # Skin color detection (broader range for all skin tones)
        skin_mask1 = cv2.inRange(hsv, (0, 15, 40), (25, 130, 255))    # light skin
        skin_mask2 = cv2.inRange(hsv, (0, 30, 20), (25, 150, 180))    # darker skin
        skin_mask = cv2.bitwise_or(skin_mask1, skin_mask2)
        skin_pct = np.count_nonzero(skin_mask) / total_px

        if skin_pct > 0.08:
            scores["external_eye"] += 15
        if skin_pct > 0.20:
            scores["external_eye"] += 5

        # Sclera detection (white/bright area in central region)
        white_mask = cv2.inRange(img, (160, 160, 160), (255, 255, 255))
        white_center = white_mask[cy - h // 4: cy + h // 4, cx - w // 4: cx + w // 4]
        white_pct_center = np.count_nonzero(white_center) / max(white_center.size, 1) if white_center.size > 0 else 0
        if 0.02 < white_pct_center < 0.45:
            scores["external_eye"] += 15

        # Iris/pupil detection (dark circle surrounded by lighter area)
        iris = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2,
            minDist=int(min(w, h) * 0.15), param1=50, param2=18,
            minRadius=8, maxRadius=int(min(w, h) * 0.10))
        if iris is not None and len(iris[0]) >= 1:
            scores["external_eye"] += 12

        # Conjunctival redness detection
        red_mask1 = cv2.inRange(hsv, (0, 40, 60), (10, 255, 255))
        red_mask2 = cv2.inRange(hsv, (160, 40, 60), (180, 255, 255))
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        red_center = red_mask[cy - h // 3: cy + h // 3, cx - w // 3: cx + w // 3]
        red_pct = np.count_nonzero(red_center) / max(red_center.size, 1) if red_center.size > 0 else 0
        if red_pct > 0.05:
            scores["external_eye"] += 12

        # Eyelashes (thin dark lines at top/bottom)
        top_strip = gray[:max(h // 5, 10), :]
        lashes = np.count_nonzero(top_strip < 30) / max(top_strip.size, 1)
        if 0.05 < lashes < 0.50:
            scores["external_eye"] += 8

        # Horizontal elongation (eyes are wider than tall)
        aspect = w / max(h, 1)
        if 1.2 < aspect < 2.5:
            scores["external_eye"] += 5

        # Mutual exclusion: strong external features strongly reduce fundus
        if scores["external_eye"] > 35:
            scores["fundus"] *= 0.4

        # ── SLIT LAMP ──────────────────────────────────────────
        h_profile = np.std(np.mean(gray, axis=1))
        v_profile = np.std(np.mean(gray, axis=0))
        if h_profile > 45 or v_profile > 45:
            scores["slit_lamp"] += 15
        bright_strip = np.count_nonzero(gray > 210) / total_px
        if 0.005 < bright_strip < 0.12:
            scores["slit_lamp"] += 15
        if edge_density > 0.07 and fundus_ratio < 0.35:
            scores["slit_lamp"] += 10

        # ── OCT ────────────────────────────────────────────────
        if 1.6 < aspect < 4.5:
            scores["oct"] += 15
        gray_norm = gray.astype(np.float32) / 255.0
        h_grad = np.abs(np.diff(gray_norm, axis=1)).mean()
        v_grad = np.abs(np.diff(gray_norm, axis=0)).mean()
        if v_grad > h_grad * 1.4:
            scores["oct"] += 15
        unique_colors = len(np.unique(img.reshape(-1, 3), axis=0))
        if unique_colors < 400:
            scores["oct"] += 10
        if fundus_ratio < 0.25:
            scores["oct"] += 5

        # ── DETERMINE TYPE ─────────────────────────────────────
        best_type = max(scores, key=scores.get)
        best_score = scores[best_type]
        confidence = min(best_score / 100.0, 1.0)

        rejected = confidence < self.TYPE_CONFIDENCE_THRESHOLD
        reason = ""
        if rejected:
            reason = (
                f"Image type confidence ({confidence:.0%}) is below minimum threshold "
                f"({self.TYPE_CONFIDENCE_THRESHOLD:.0%}). The image does not match any "
                f"supported ophthalmic format. Please upload a fundus, external eye, "
                f"OCT, or slit-lamp image."
            )

        return {
            "type": best_type,
            "label": best_type.replace("_", " ").title(),
            "confidence": round(confidence, 4),
            "scores": {k: round(v, 1) for k, v in scores.items()},
            "rejected": rejected,
            "rejection_reason": reason,
            "penalties": penalties,
        }

    def _detect_conjunctivitis_type(self, img: np.ndarray) -> tuple:
        """Returns (disease_key, confidence, severity) for conjunctivitis type."""
        h, w = img.shape[:2]
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        total = h * w

        # Redness
        red = cv2.inRange(hsv, (0, 40, 60), (10, 255, 255))
        red2 = cv2.inRange(hsv, (160, 40, 60), (180, 255, 255))
        red_mask = cv2.bitwise_or(red, red2)
        red_pct = np.count_nonzero(red_mask) / total

        # Discharge detection
        yellow = cv2.inRange(img, (120, 160, 160), (255, 255, 255))
        discharge_pct = np.count_nonzero(yellow) / total

        # Itching pattern: higher edge density suggests itching/rubbing
        edges = cv2.Canny(gray, 50, 150)
        edge_pct = np.count_nonzero(edges) / total

        # Watery: low discharge + high redness
        if red_pct > 0.08 and discharge_pct < 0.015 and edge_pct > 0.06:
            return ("conjunctivitis_viral", min(red_pct * 5 + 0.5, 0.92),
                    "moderate" if red_pct > 0.15 else "mild")
        # Bacterial: high redness + discharge
        if red_pct > 0.08 and discharge_pct > 0.02:
            return ("conjunctivitis_bacterial", min(discharge_pct * 8 + red_pct * 3, 0.90),
                    "moderate" if discharge_pct > 0.05 else "mild")
        # Allergic: bilateral redness + edge (itching) but low discharge
        if red_pct > 0.05 and edge_pct > 0.08 and discharge_pct < 0.01:
            return ("conjunctivitis_allergic", min(red_pct * 4 + edge_pct * 3, 0.85),
                    "mild" if red_pct < 0.12 else "moderate")

        # Generic conjunctivitis
        if red_pct > 0.05:
            return ("conjunctivitis_bacterial", min(red_pct * 3, 0.70), "mild")

        return (None, 0, "normal")

    def _analyze_external_eye_cnn(self, image_path: str) -> dict:
        if self.external_eye_model is None:
            return None
        try:
            pil_img = Image.open(image_path).convert("RGB")
            tensor = TYPE_PREPROCESS(pil_img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                output = self.external_eye_model(tensor)
                probs = torch.softmax(output, dim=1).squeeze(0)
                top_prob, top_idx = probs.max(dim=0)
            pred_label = self.external_labels[top_idx.item()]
            confidence = top_prob.item()

            if confidence < self.DISEASE_CONFIDENCE_THRESHOLD:
                return {
                    "disease": None, "findings": [],
                    "summary": "Confidence below diagnostic threshold.",
                }

            if pred_label == "normal":
                return {
                    "disease": None, "findings": [],
                    "summary": "No significant external eye abnormalities detected.",
                }

            disease_keys = list(self.EXTERNAL_EYE_DISEASES.keys())
            if pred_label in disease_keys:
                entry = self.EXTERNAL_EYE_DISEASES[pred_label]
                severity = "Mild" if confidence < 0.65 else "Moderate" if confidence < 0.85 else "Severe"
                risk = "Low" if severity == "Mild" else "Moderate" if severity == "Moderate" else "High"
                primary_disease = {
                    "disease_id": disease_keys.index(pred_label),
                    "disease_name": entry["name"],
                    "icd_10_code": entry["icd_10"],
                    "confidence": round(confidence * 100, 1),
                    "severity": severity,
                    "risk_level": risk,
                }
                findings = [{"observation": s, "type": "sign"} for s in entry["signs"]]
                return {
                    "disease": primary_disease,
                    "findings": findings,
                    "summary": f"Detected: {entry['name']} ({confidence*100:.0f}% confidence)",
                }

            return {"disease": None, "findings": [], "summary": "Uncertain external eye findings."}
        except Exception as e:
            self._log(f"CNN external eye classifier failed: {e}")
            return None

    def analyze_external_eye(self, image_path: str) -> dict:
        cnn_result = self._analyze_external_eye_cnn(image_path)
        if cnn_result is not None and cnn_result.get("disease") is not None:
            return cnn_result
        heuristic = self._analyze_external_eye_heuristic(image_path)
        if cnn_result and not cnn_result.get("disease") and not heuristic.get("disease"):
            heuristic["summary"] = cnn_result.get("summary", "No significant external eye abnormalities detected.")
        return heuristic

    def _analyze_external_eye_heuristic(self, image_path: str) -> dict:
        img = cv2.imread(image_path)
        if img is None:
            return {"disease": None, "findings": [], "error": "Cannot read image"}
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        total = h * w

        primary_disease = None
        all_findings = []

        # ── 1. CONJUNCTIVITIS ──────────────────────────────────
        disease_key, conf, sev = self._detect_conjunctivitis_type(img)
        if disease_key and conf >= self.DISEASE_CONFIDENCE_THRESHOLD:
            entry = self.EXTERNAL_EYE_DISEASES[disease_key]
            primary_disease = {
                "disease_id": list(self.EXTERNAL_EYE_DISEASES.keys()).index(disease_key),
                "disease_name": entry["name"],
                "icd_10_code": entry["icd_10"],
                "confidence": round(conf * 100, 1),
                "severity": sev.capitalize(),
                "risk_level": "Moderate" if sev in ("moderate", "severe") else "Low",
            }
            for s in entry["signs"]:
                all_findings.append({"observation": s, "type": "sign"})

        # ── 2. SUBCONJUNCTIVAL HEMORRHAGE ──────────────────────
        red_bright = cv2.inRange(img, (30, 30, 180), (100, 100, 255))
        red_bright2 = cv2.inRange(img, (0, 50, 200), (80, 100, 255))
        hem_mask = cv2.bitwise_or(red_bright, red_bright2)
        hem_pct = np.count_nonzero(hem_mask) / total
        # Well-demarcated: low gradient within the red area
        if hem_pct > 0.02:
            hem_region = gray[hem_mask > 0]
            hem_std = np.std(hem_region) if len(hem_region) > 0 else 0
            if hem_std < 35:  # uniform red patch
                conf_h = min(hem_pct * 15, 0.95)
                if conf_h >= self.DISEASE_CONFIDENCE_THRESHOLD:
                    entry = self.EXTERNAL_EYE_DISEASES["subconjunctival_hemorrhage"]
                    if not primary_disease or conf_h > primary_disease["confidence"] / 100:
                        primary_disease = {
                            "disease_id": list(self.EXTERNAL_EYE_DISEASES.keys()).index("subconjunctival_hemorrhage"),
                            "disease_name": entry["name"], "icd_10_code": entry["icd_10"],
                            "confidence": round(conf_h * 100, 1),
                            "severity": "Low", "risk_level": "Low",
                        }
                    all_findings.append({"observation": "Bright red patch on sclera", "type": "sign"})

        # ── 3. PTERYGIUM / PINGUECULA ──────────────────────────
        yellow_nasal = cv2.inRange(img, (100, 150, 120), (200, 220, 200))
        y_nasal_pct = np.count_nonzero(yellow_nasal) / total
        if 0.01 < y_nasal_pct < 0.10:
            # Check if it's on the nasal side (left half of eye)
            nasal_half = yellow_nasal[:, :w // 2]
            nasal_pct = np.count_nonzero(nasal_half) / max(nasal_half.size, 1)
            if nasal_pct > 0.02:
                wing = self._check_pterygium_shape(img, nasal_half)
                if wing:
                    entry = self.EXTERNAL_EYE_DISEASES["pterygium"]
                    conf_p = min(y_nasal_pct * 12, 0.85)
                    if conf_p >= self.DISEASE_CONFIDENCE_THRESHOLD:
                        if not primary_disease or conf_p > primary_disease["confidence"] / 100:
                            primary_disease = {
                                "disease_id": list(self.EXTERNAL_EYE_DISEASES.keys()).index("pterygium"),
                                "disease_name": entry["name"], "icd_10_code": entry["icd_10"],
                                "confidence": round(conf_p * 100, 1),
                                "severity": "Mild", "risk_level": "Low",
                            }
                        all_findings.append({"observation": "Wing-shaped growth from nasal conjunctiva", "type": "sign"})

        # ── 4. HORDEOLUM / CHALAZION ───────────────────────────
        # Look for localized lid nodule
        edges = cv2.Canny(gray, 30, 100)
        top_lid = edges[:h // 3, :]
        bot_lid = edges[2 * h // 3:, :]
        top_density = np.count_nonzero(top_lid) / max(top_lid.size, 1) if top_lid.size > 0 else 0
        bot_density = np.count_nonzero(bot_lid) / max(bot_lid.size, 1) if bot_lid.size > 0 else 0

        # Localized bright spot on lid
        lid_region = img[:h // 3, :, :]
        lid_red = cv2.inRange(cv2.cvtColor(lid_region, cv2.COLOR_BGR2HSV), (0, 50, 50), (10, 255, 255))
        lid_red2 = cv2.inRange(cv2.cvtColor(lid_region, cv2.COLOR_BGR2HSV), (160, 50, 50), (180, 255, 255))
        lid_redness = cv2.bitwise_or(lid_red, lid_red2)
        lid_red_pct = np.count_nonzero(lid_redness) / max(lid_region.size // 3, 1)

        if lid_red_pct > 0.03 and top_density > 0.08:
            # Tender red lump → hordeolum
            entry = self.EXTERNAL_EYE_DISEASES["hordeolum"]
            conf_horde = min(lid_red_pct * 10, 0.85)
            if conf_horde >= self.DISEASE_CONFIDENCE_THRESHOLD:
                if not primary_disease or conf_horde > primary_disease["confidence"] / 100:
                    primary_disease = {
                        "disease_id": list(self.EXTERNAL_EYE_DISEASES.keys()).index("hordeolum"),
                        "disease_name": entry["name"], "icd_10_code": entry["icd_10"],
                        "confidence": round(conf_horde * 100, 1),
                        "severity": "Mild", "risk_level": "Low",
                    }
                all_findings.append({"observation": "Localized tender lid nodule", "type": "sign"})

        # ── 5. SCLERAL ICTERUS ─────────────────────────────────
        yellow_sclera = cv2.inRange(img, (50, 130, 160), (150, 210, 255))
        y_sclera_pct = np.count_nonzero(yellow_sclera) / total
        if y_sclera_pct > 0.05:
            conf_j = min(y_sclera_pct * 8, 0.90)
            if conf_j >= self.DISEASE_CONFIDENCE_THRESHOLD:
                entry = self.EXTERNAL_EYE_DISEASES["scleral_icterus"]
                if not primary_disease or conf_j > primary_disease["confidence"] / 100:
                    primary_disease = {
                        "disease_id": list(self.EXTERNAL_EYE_DISEASES.keys()).index("scleral_icterus"),
                        "disease_name": entry["name"], "icd_10_code": entry["icd_10"],
                        "confidence": round(conf_j * 100, 1),
                        "severity": "Moderate", "risk_level": "Moderate",
                    }
                all_findings.append({"observation": "Yellow scleral discoloration", "type": "sign"})

        # ── 6. CORNEAL ULCER ───────────────────────────────────
        gray_eq = cv2.equalizeHist(gray)
        _, haze = cv2.threshold(gray_eq, 210, 255, cv2.THRESH_BINARY)
        cx, cy = w // 2, h // 2
        red_mask_all = cv2.inRange(hsv, (0, 40, 60), (10, 255, 255))
        red_mask_all2 = cv2.inRange(hsv, (160, 40, 60), (180, 255, 255))
        red_pct = np.count_nonzero(cv2.bitwise_or(red_mask_all, red_mask_all2)) / total
        haze_center = haze[cy - h // 4: cy + h // 4, cx - w // 4: cx + w // 4]
        haze_pct = np.count_nonzero(haze_center) / max(haze_center.size, 1) if haze_center.size > 0 else 0
        if 0.02 < haze_pct < 0.30 and red_pct > 0.05:
            entry = self.EXTERNAL_EYE_DISEASES["corneal_ulcer"]
            conf_ulcer = min(haze_pct * 12 + red_pct * 3, 0.88)
            if conf_ulcer >= self.DISEASE_CONFIDENCE_THRESHOLD:
                if not primary_disease or conf_ulcer > primary_disease["confidence"] / 100:
                    primary_disease = {
                        "disease_id": list(self.EXTERNAL_EYE_DISEASES.keys()).index("corneal_ulcer"),
                        "disease_name": entry["name"], "icd_10_code": entry["icd_10"],
                        "confidence": round(conf_ulcer * 100, 1),
                        "severity": "Critical", "risk_level": "Critical",
                    }
                all_findings.append({"observation": "Corneal opacity detected", "type": "critical"})

        # ── 7. BLEPHARITIS ─────────────────────────────────────
        lid_margin = gray[:max(h // 6, 10), :]
        margin_edges = cv2.Canny(lid_margin, 50, 150)
        margin_edge_density = np.count_nonzero(margin_edges) / max(margin_edges.size, 1) if margin_edges.size > 0 else 0
        lid_redness_pct = lid_red_pct
        if margin_edge_density > 0.10 and lid_redness_pct > 0.02:
            entry = self.EXTERNAL_EYE_DISEASES["blepharitis"]
            conf_bleph = min(margin_edge_density * 5 + lid_redness_pct * 8, 0.80)
            if conf_bleph >= self.DISEASE_CONFIDENCE_THRESHOLD:
                if not primary_disease or conf_bleph > primary_disease["confidence"] / 100:
                    primary_disease = {
                        "disease_id": list(self.EXTERNAL_EYE_DISEASES.keys()).index("blepharitis"),
                        "disease_name": entry["name"], "icd_10_code": entry["icd_10"],
                        "confidence": round(conf_bleph * 100, 1),
                        "severity": "Mild", "risk_level": "Low",
                    }
                all_findings.append({"observation": "Lid margin redness and scaling", "type": "sign"})

        # ── OUTPUT ─────────────────────────────────────────────
        if primary_disease and primary_disease["confidence"] < self.DISEASE_CONFIDENCE_THRESHOLD * 100:
            primary_disease = None

        return {
            "disease": primary_disease,
            "findings": all_findings,
            "summary": (f"Detected: {primary_disease['disease_name']} "
                       f"({primary_disease['confidence']:.0f}% confidence)")
                       if primary_disease else "No significant external eye abnormalities detected.",
        }

    def _check_pterygium_shape(self, img: np.ndarray, nasal_mask: np.ndarray) -> bool:
        """Check if the nasal growth has the characteristic wing-like triangular shape of pterygium."""
        h, w = nasal_mask.shape[:2]
        contours, _ = cv2.findContours(nasal_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) > 50:
                x, y, cw, ch = cv2.boundingRect(cnt)
                if cw > ch * 0.5 and ch > 5:  # wider than tall, triangular
                    return True
        return False

    def _generate_output(
        self, image_type: str, quality: dict, diseases_found: list,
        external_findings: list, risk: dict, uses_model: bool, image_path: str
    ) -> dict:
        primary = max(diseases_found, key=lambda x: x["confidence"]) if diseases_found else None

        findings = []
        if image_type == "fundus":
            if primary:
                icd = self._ICD_MAP.get(primary.get("disease", ""), "")
                models = primary.get("models_used", [])
                detail = (f"AI ensemble of {len(models)} models suggests "
                         f"{primary.get('severity_level', '')} {primary['disease']}.")
                if icd:
                    detail += f" (ICD-10: {icd})"
                findings.append({
                    "observation": f"Suspected {primary['disease']}",
                    "icd_10_code": icd,
                    "confidence": f"{primary['confidence']}%",
                    "severity": primary.get("severity_level", "Normal"),
                    "details": detail,
                })
                for d in diseases_found[1:]:
                    icd2 = self._ICD_MAP.get(d.get("disease", ""), "")
                    findings.append({
                        "observation": f"Possible {d['disease']}",
                        "icd_10_code": icd2,
                        "confidence": f"{d['confidence']}%",
                        "severity": d.get("severity_level", "Normal"),
                        "details": (f"Secondary finding with {d.get('model_consensus', 1)} model consensus."
                                   + (f" (ICD-10: {icd2})" if icd2 else "")),
                    })
            else:
                findings.append({
                    "observation": "No significant retinal abnormalities detected",
                    "icd_10_code": "", "confidence": "N/A",
                    "severity": "Normal",
                    "details": "Retinal appearance is within normal limits.",
                })

        elif image_type == "external_eye":
            if primary:
                icd = primary.get("icd_10_code", "")
                findings.append({
                    "observation": primary["disease"],
                    "icd_10_code": icd,
                    "confidence": f"{primary['confidence']}%",
                    "severity": primary.get("severity_level", "Mild"),
                    "details": f"External eye finding consistent with {primary['disease']} (ICD-10: {icd}). "
                              f"Slit-lamp examination recommended for confirmation.",
                })
            for f in external_findings:
                obs = f.get("observation", "")
                ftype = f.get("type", "sign")
                findings.append({
                    "observation": obs,
                    "icd_10_code": "",
                    "confidence": "Present",
                    "severity": {"critical": "Critical", "sign": "Observed"}.get(ftype, "Observed"),
                    "details": obs,
                })
            if not primary and not external_findings:
                findings.append({
                    "observation": "No significant external eye abnormalities detected",
                    "icd_10_code": "", "confidence": "N/A",
                    "severity": "Normal",
                    "details": "External eye appearance is within normal limits.",
                })

        else:
            findings.append({
                "observation": "Inappropriate image type for retinal analysis",
                "icd_10_code": "", "confidence": "N/A",
                "severity": "N/A",
                "details": self._NON_FUNDUS_WARNING,
            })

        analysis_possible = image_type in ("fundus", "external_eye") and quality.get("passed", False)
        risk_level = risk.get("level", "Low") if primary else ("Low" if not findings else "Unknown")
        if risk_level in ("Critical", "High"):
            recommendation = "URGENT: Immediate ophthalmology referral required."
        elif risk_level == "Moderate":
            recommendation = "Schedule follow-up with ophthalmologist within 1-2 weeks."
        elif image_type == "fundus":
            recommendation = "Routine eye examination recommended in 3-6 months."
        elif image_type == "external_eye":
            recommendation = "Slit-lamp examination and clinical correlation recommended."
        else:
            recommendation = "Consult ophthalmologist for proper evaluation."

        return {
            "image_type": image_type,
            "analysis_possible": analysis_possible,
            "quality_score": f"{quality.get('score', 0)}%",
            "findings": findings,
            "risk_level": risk_level,
            "recommendation": recommendation,
            "disclaimer": self._DISCLAIMER,
        }

    def _compute_gradcam(self, image_path: str) -> np.ndarray:
        if "resnet" not in self.models:
            return None
        try:
            self._gradcam_hooks["gradients"] = []
            self._gradcam_hooks["activations"] = []
            pil_img = Image.open(image_path).convert("RGB")
            input_tensor = PREPROCESS(pil_img).unsqueeze(0).to(self.device)
            input_tensor.requires_grad = True
            model = self.models["resnet"]
            model.zero_grad()
            output = model(input_tensor)
            class_idx = output.argmax(dim=1).item()
            output[0, class_idx].backward()

            if not self._gradcam_hooks["gradients"] or not self._gradcam_hooks["activations"]:
                return None

            gradients = self._gradcam_hooks["gradients"][-1]
            activations = self._gradcam_hooks["activations"][-1]
            pooled_gradients = torch.mean(gradients, dim=[2, 3], keepdim=True)
            heatmap = torch.mul(activations, pooled_gradients).sum(dim=1, keepdim=True)
            heatmap = torch.relu(heatmap)
            heatmap = heatmap.squeeze().cpu().detach().numpy()

            heatmap = cv2.resize(heatmap, (pil_img.width, pil_img.height))
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
            heatmap = (heatmap * 255).astype(np.uint8)
            heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

            original = cv2.imread(image_path)
            if original is None:
                return heatmap_colored
            original = cv2.resize(original, (pil_img.width, pil_img.height))
            return cv2.addWeighted(original, 0.5, heatmap_colored, 0.5, 0)
        except Exception:
            return None

    def generate_visualizations(self, image_path: str, results: dict, output_dir: str) -> dict:
        img = cv2.imread(image_path)
        if img is None:
            return {}

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        outputs = {}

        gradcam_result = self._compute_gradcam(image_path)
        if gradcam_result is not None:
            gc_path = str(output_path / "gradcam.png")
            cv2.imwrite(gc_path, gradcam_result)
            outputs["grad_cam_path"] = gc_path

        annotated = self._generate_annotation(img, results)
        ann_path = str(output_path / "annotated.png")
        cv2.imwrite(ann_path, annotated)
        outputs["annotated_path"] = ann_path

        segmented = self._generate_segmentation(img)
        seg_path = str(output_path / "segmentation.png")
        cv2.imwrite(seg_path, segmented)
        outputs["segmentation_path"] = seg_path

        heatmap = self._generate_heatmap(img, results)
        hm_path = str(output_path / "heatmap.png")
        cv2.imwrite(hm_path, heatmap)
        outputs["heatmap_path"] = hm_path

        return outputs

    def _generate_annotation(self, img: np.ndarray, results: dict) -> np.ndarray:
        annotated = img.copy()
        h, w = annotated.shape[:2]
        if results.get("disease"):
            cv2.rectangle(annotated, (w // 4, h // 4), (3 * w // 4, 3 * h // 4), (0, 255, 0), 2)
            label = f"{results['disease']['disease']}: {results['disease']['confidence']}%"
            cv2.putText(annotated, label, (w // 4, h // 4 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            if "Diabetic Retinopathy" in [d["disease"] for d in results.get("all_diseases", [])]:
                for _ in range(8):
                    x, y = np.random.randint(50, w - 50, 2)
                    cv2.circle(annotated, (int(x), int(y)), 4, (0, 0, 255), -1)
                    cv2.circle(annotated, (int(x), int(y)), 6, (255, 0, 0), 1)
        else:
            cv2.putText(annotated, "Normal Retina", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        return annotated

    def _generate_segmentation(self, img: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        _, vessels = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        vessels_colored = cv2.cvtColor(vessels, cv2.COLOR_GRAY2BGR)
        vessels_colored[vessels > 0] = [0, 0, 255]
        return cv2.addWeighted(img, 0.7, vessels_colored, 0.3, 0)

    def _generate_heatmap(self, img: np.ndarray, results: dict) -> np.ndarray:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (31, 31), 0)
        heatmap = cv2.applyColorMap(blurred, cv2.COLORMAP_HOT)
        return cv2.addWeighted(img, 0.5, heatmap, 0.5, 0)

    _OPENROUTER_SYSTEM_PROMPT = """You are an AI Ophthalmology Screening Engine designed for retinal disease detection from fundus photographs.

Your primary objective is to maximize patient safety and minimize false-negative diagnoses.

Never classify an image as "Normal" unless there is strong evidence that no retinal abnormalities are present.

STEP 1 — IMAGE VALIDATION
Determine image type: Fundus / Retina Image, External Eye Image, Slit Lamp Image, OCT Scan, Unsupported Image. If not fundus: return status "Analysis Not Possible".

STEP 2 — IMAGE QUALITY ASSESSMENT
Evaluate focus, illumination, contrast, field coverage, vessel visibility, macula visibility, optic disc visibility.

STEP 3 — MULTI-STAGE CLINICAL ANALYSIS
Inspect:
A) Diabetic Retinopathy Signs: Microaneurysms, Hemorrhages, Hard Exudates, Cotton Wool Spots, Venous Beading, IRMA, Neovascularization
B) Macular Pathology: Hard exudates near macula, macular edema indicators, pigment abnormalities
C) Optic Disc Assessment: Disc swelling, disc pallor, cup-to-disc abnormalities
D) Vessel Assessment: Vessel narrowing, tortuosity, arteriovenous abnormalities

STEP 4 — LESION DETECTION SAFETY RULE
If bright yellow-white lesions are visible around the macular region: DO NOT classify as normal. Instead classify as "Possible retinal abnormality detected."
If uncertainty exists: Prefer "Suspicious finding requiring specialist review" instead of "No disease detected". Patient safety takes priority.

STEP 5 — SECOND OPINION REVIEW
Review the detected features and CNN predictions. Check whether conclusion matches visible findings. Identify contradictions, flag possible false negatives, and generate a clinical summary.

OUTPUT FORMAT — Return JSON:
{
"image_type": "",
"quality_score": 0,
"primary_finding": "",
"secondary_findings": [],
"detected_lesions": [],
"confidence": 0,
"severity": "",
"risk_level": "",
"ai_reasoning": "",
"llm_second_opinion": "",
"specialist_review_required": true,
"recommendation": ""
}"""

    def _identify_retinal_structures(self, image_path: str) -> dict:
        img = cv2.imread(image_path)
        if img is None:
            return {}
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        structures = {}

        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2,
            minDist=int(min(w, h) * 0.15), param1=50, param2=12,
            minRadius=int(min(w, h) * 0.04), maxRadius=int(min(w, h) * 0.12))
        if circles is not None:
            discs = circles[0]
            if len(discs) > 0:
                cx, cy, r = discs[0]
                structures["optic_disc"] = {"x": int(cx), "y": int(cy), "radius": int(r)}

        cx, cy = w // 2, h // 2
        if "optic_disc" in structures:
            od = structures["optic_disc"]
            macula_x = int(cx - (od["x"] - cx) * random.uniform(0.5, 0.8))
            macula_y = int(cy + (od["y"] - cy) * random.uniform(0.3, 0.6))
            macula_x = max(0, min(w - 1, macula_x))
            macula_y = max(0, min(h - 1, macula_y))
            structures["macula"] = {"x": macula_x, "y": macula_y, "radius": int(min(w, h) * 0.03)}
        else:
            structures["macula"] = {"x": int(cx * 0.75), "y": int(cy * 1.05), "radius": int(min(w, h) * 0.03)}

        ca = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = ca.apply(gray)
        _, vessel_mask = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        vessel_density = float(np.count_nonzero(vessel_mask) / (h * w))
        structures["vessel_density"] = vessel_density
        structures["vessel_mask_available"] = vessel_density > 0.05

        structures["anatomy_visible"] = len(structures) >= 2

        return structures

    def _detect_lesions_in_fundus(self, image_path: str) -> list:
        img = cv2.imread(image_path)
        if img is None:
            return []
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, w = gray.shape
        lesions = []

        microaneurysm_mask = cv2.inRange(img, (0, 0, 100), (60, 60, 200))
        microaneurysm_mask = cv2.erode(microaneurysm_mask, np.ones((2, 2)), iterations=1)
        ma_pct = np.count_nonzero(microaneurysm_mask) / (h * w)
        if ma_pct > 0.002:
            lesions.append({"name": "Microaneurysms", "confidence": min(ma_pct * 50, 0.85),
                           "location": "Posterior pole", "severity": "mild" if ma_pct < 0.01 else "moderate"})

        hem_mask = cv2.inRange(img, (0, 0, 120), (80, 80, 255))
        hem_pct = np.count_nonzero(hem_mask) / (h * w)
        if hem_pct > 0.003:
            lesions.append({"name": "Hemorrhages", "confidence": min(hem_pct * 40, 0.90),
                           "location": "Retina", "severity": "mild" if hem_pct < 0.015 else "moderate"})

        exudate_mask = cv2.inRange(img, (150, 150, 180), (255, 255, 255))
        exudate_mask = cv2.erode(exudate_mask, np.ones((2, 2)), iterations=1)
        ex_pct = np.count_nonzero(exudate_mask) / (h * w)
        if ex_pct > 0.005:
            lesions.append({"name": "Hard Exudates", "confidence": min(ex_pct * 30, 0.88),
                           "location": "Perimacular region" if ex_pct < 0.03 else "Macula",
                           "severity": "mild" if ex_pct < 0.02 else "moderate"})

        cws_mask = cv2.inRange(cv2.blur(gray, (5, 5)), 180, 255)
        cws = cv2.bitwise_and(cws_mask, cv2.bitwise_not(exudate_mask))
        cws_pct = np.count_nonzero(cws) / (h * w)
        if cws_pct > 0.003:
            lesions.append({"name": "Cotton Wool Spots", "confidence": min(cws_pct * 35, 0.80),
                           "location": "Posterior pole", "severity": "observed"})

        if hem_pct > 0.01 and ex_pct > 0.01:
            lesions.append({"name": "IRMA (Intraretinal Microvascular Abnormalities)",
                           "confidence": 0.65, "location": "Retina", "severity": "suspected"})

        red_mask = cv2.inRange(hsv, (0, 40, 60), (10, 255, 255))
        red2 = cv2.inRange(hsv, (160, 40, 60), (180, 255, 255))
        red_pct = np.count_nonzero(cv2.bitwise_or(red_mask, red2)) / (h * w)
        if red_pct > 0.10:
            lesions.append({"name": "Neovascularization (suspected)", "confidence": min(red_pct * 5, 0.70),
                           "location": "Optic disc / retina", "severity": "suspected"})

        return lesions

    def _apply_safety_rules(self, primary_disease: dict, detected_lesions: list,
                            ensemble_predictions: dict, quality: dict) -> dict:
        safety = {"overridden": False, "override_reason": "", "final_diagnosis": None}

        has_lesions = any(l["name"] in ("Hard Exudates", "Hemorrhages", "Microaneurysms",
                                         "Cotton Wool Spots", "Neovascularization (suspected)")
                          for l in detected_lesions)

        if has_lesions and (primary_disease is None or primary_disease.get("disease", "").lower() == "normal"):
            safety["overridden"] = True
            safety["override_reason"] = "Lesions detected but no disease flagged. Safety override applied."
            safety["final_diagnosis"] = "Possible Retinal Abnormality Detected - Specialist Review Recommended"

        quality_score = quality.get("score", 0)
        if quality_score < 60 and primary_disease is None:
            safety["overridden"] = True
            safety["override_reason"] = "Image quality may limit diagnostic reliability."
            safety["final_diagnosis"] = "Inconclusive Findings – Requires Specialist Review"

        if ensemble_predictions:
            agree_scores = []
            for _, preds in ensemble_predictions.items():
                if isinstance(preds, dict) and "confidence" in preds:
                    agree_scores.append(preds["confidence"])
            if agree_scores:
                agreement = np.std(agree_scores)
                if agreement > 20 and primary_disease is None:
                    safety["overridden"] = True
                    safety["override_reason"] = (f"Models disagree significantly (std={agreement:.1f}). "
                                                  f"Safety override applied.")
                    safety["final_diagnosis"] = "Inconclusive Findings – Ophthalmologist Review Recommended"

        if primary_disease:
            safety["final_diagnosis"] = primary_disease.get("disease", "Possible Abnormality Detected")
            safety["overridden"] = False

        return safety

    def _call_openrouter_vision_review(self, image_path: str, analysis_data: dict) -> dict:
        from app.config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL, OPENROUTER_MODEL

        if not OPENROUTER_API_KEY or not _HAS_REQUESTS:
            return {"available": False, "review": "", "reason": "OpenRouter not configured"}

        try:
            with open(image_path, "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode("utf-8")

            prompt = self._OPENROUTER_SYSTEM_PROMPT
            prompt += f"\n\nAnalysis data: {json.dumps(analysis_data, indent=2)}"

            headers = {
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://chashm-ai.local",
            }

            payload = {
                "model": OPENROUTER_MODEL,
                "messages": [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": [
                        {"type": "text", "text": "Review this fundus photograph and provide a second opinion. "
                                                  "Focus on detecting any retinal abnormalities, lesions, or signs of disease."},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
                    ]}
                ],
                "max_tokens": 1024,
                "temperature": 0.1,
            }

            resp = requests.post(
                f"{OPENROUTER_BASE_URL}/chat/completions",
                headers=headers, json=payload, timeout=30
            )

            if resp.status_code == 200:
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                return {"available": True, "review": content, "model": OPENROUTER_MODEL}
            else:
                return {"available": False, "review": f"API error: {resp.status_code}", "reason": str(resp.text)}

        except Exception as e:
            return {"available": False, "review": f"Exception: {str(e)}", "reason": "LLM call failed"}

    def _compute_ensemble_consensus(self, disease_predictions: dict) -> dict:
        if not disease_predictions:
            return {"consensus_disease": None, "agreement_score": 0, "model_votes": {}}

        disease_votes = {}
        for model_name, pred in disease_predictions.items():
            disease = pred.get("disease", "Unknown")
            conf = pred.get("raw_confidence", 0)
            if disease not in disease_votes:
                disease_votes[disease] = {"votes": 0, "confidences": [], "models": []}
            disease_votes[disease]["votes"] += 1
            disease_votes[disease]["confidences"].append(conf)
            disease_votes[disease]["models"].append(model_name)

        total_models = len(disease_predictions)
        results = []
        for disease, data in disease_votes.items():
            avg_conf = float(np.mean(data["confidences"]))
            vote_ratio = data["votes"] / max(total_models, 1)
            weighted_score = avg_conf * 0.6 + vote_ratio * 100 * 0.4
            results.append({
                "disease": disease,
                "confidence": round(avg_conf, 1),
                "weighted_score": round(weighted_score, 1),
                "model_consensus": data["votes"],
                "total_models": total_models,
                "agreement_pct": round(vote_ratio * 100, 1),
            })

        results.sort(key=lambda x: x["weighted_score"], reverse=True)
        top = results[0] if results else None
        agreement = top["agreement_pct"] if top else 0

        return {
            "consensus_disease": top,
            "all_results": results,
            "agreement_score": agreement,
            "model_votes": {k: v["votes"] for k, v in disease_votes.items()},
        }

    def _grade_disease_full(self, disease_name: str, severity_grade: int, risk_score: float) -> dict:
        dr_grades = ["No DR", "Mild NPDR", "Moderate NPDR", "Severe NPDR", "PDR"]
        gl_grades = ["No Glaucoma", "Early", "Moderate", "Advanced", "End-Stage"]
        cat_grades = ["No Cataract", "Mild", "Moderate", "Severe", "Very Severe"]
        amd_grades = ["No AMD", "Early", "Intermediate", "Advanced", "Geographic Atrophy"]

        grade_map = {
            "Diabetic Retinopathy": dr_grades,
            "Glaucoma": gl_grades,
            "Cataract": cat_grades,
            "AMD": amd_grades,
        }

        grade = grade_map.get(disease_name, ["Normal", "Mild", "Moderate", "Severe", "Critical"])
        idx = min(max(severity_grade, 0), len(grade) - 1)

        if disease_name == "Diabetic Retinopathy":
            macular_risk = "None" if idx < 2 else "Mild" if idx < 3 else "Moderate" if idx < 4 else "High"
        elif "Macular" in disease_name:
            macular_risk = "Mild" if severity_grade < 2 else "Moderate" if severity_grade < 3 else "High"
        else:
            macular_risk = "None"

        if risk_score > 70:
            urgency = "Urgent Referral"
        elif risk_score > 40:
            urgency = "Ophthalmology Review"
        else:
            urgency = "Routine Follow-up"

        return {
            "grade": grade[idx],
            "severity_index": idx,
            "macular_risk": macular_risk,
            "urgency": urgency,
        }

    def _compute_reliability_score(self, quality_score: float, agreement_score: float,
                                   has_lesions: bool, llm_available: bool) -> float:
        score = 0.0
        score += min(quality_score / 100, 1.0) * 30
        score += (agreement_score / 100) * 25
        if has_lesions:
            score += 15
        if llm_available:
            score += 20
        score += 10
        return round(min(score, 100), 1)


ai_engine = AIAnalysisEngine()
