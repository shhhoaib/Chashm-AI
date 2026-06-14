# Chashm AI

**AI-Powered Eye Disease Screening Platform**

Chashm AI is a desktop application for ophthalmic screening that uses a multi-model AI ensemble to detect eye diseases from fundus photographs and external eye images. It features an 11-stage clinical pipeline, professional PDF reporting, patient management, and optional LLM-based second-opinion review via OpenRouter.

---

## Features

### AI Screening Engine
- **11-Stage Clinical Pipeline**: Image validation, quality grading, retinal structure identification, lesion detection, multi-model ensemble, safety rules, and consensus decision
- **Multi-Model Ensemble**: EfficientNetV2-S, ConvNeXt Tiny, ResNet18, MobileNetV3-Small
- **External Eye Analysis**: Detects 15 external eye conditions (conjunctivitis, pterygium, hordeolum, etc.)
- **Image Type Classification**: CNN-based classifier to distinguish fundus, external eye, slit-lamp, OCT, and unknown images
- **Safety-First Design**: Never outputs "Normal" when lesions are detected; always recommends specialist review when uncertain
- **Explainable AI**: Grad-CAM heatmaps, lesion annotations, segmentation maps

### Patient & Scan Management
- **Doctor Authentication**: Login/registration with JWT tokens and password hashing
- **Patient Records**: Add, edit, search, and manage patients with medical history
- **Scan Records**: Upload images, run AI analysis, store results with quality scores and confidence metrics
- **Recent Scans Dashboard**: Quick access to recent patient scans

### Reporting
- **Professional PDF Reports**: Clinical-grade layout with hospital branding
- **ICD-10 Coding**: Standard medical codes for all detected conditions
- **Severity & Risk Badges**: Color-coded severity levels
- **Dietary & Lifestyle Recommendations**: Condition-specific advice for 18+ eye diseases
- **Plain-Language Explanations**: Patient-friendly "What This Means" section
- **Barcode & QR Code**: Report verification
- **Doctor Signature & Stamp**: Professional footer

### AI Training
- **Synthetic Data Generation**: Generates realistic fundus and external eye images for training
- **Model Fine-Tuning**: Retrain models from the UI with custom parameters
- **Progress Tracking**: Real-time training progress, accuracy metrics, and logs

### User Interface
- **Dark Theme**: Modern dark UI with professional color scheme
- **Sidebar Navigation**: Dashboard, Patients, Doctors, Scan Analysis, Reports, AI Training, Settings, Logs
- **Login Dialog**: Secure authentication with hospital branding
- **Model Loading Dialog**: Animated loading with progress tracking

---

## Architecture

### 11-Stage AI Pipeline

| Stage | Component | Description |
|-------|-----------|-------------|
| 1 | Image Validation | CNN + heuristic image type classification |
| 2 | Quality Assessment | Focus, brightness, contrast, noise, completeness scoring |
| 3 | Structure Identification | Optic disc, macula, vessel detection via HoughCircles + CLAHE |
| 4 | Lesion Detection | Microaneurysms, hemorrhages, exudates, cotton wool spots, IRMA, NV |
| 5 | Ensemble Inference | 4 CNN models with weighted voting |
| 6 | Safety Rules | Never classify as normal if lesions or model disagreement exist |
| 7 | LLM Second Opinion | Optional OpenRouter vision LLM review (Qwen 2.5 VL) |
| 8 | Consensus Decision | Weighted scoring: 60% confidence + 40% model agreement |
| 9 | Disease Grading | NPDR/PDR staging, macular risk, urgency level |
| 10 | Reliability Score | Weighted: quality (30%) + agreement (25%) + lesions (15%) + LLM (20%) + baseline (10%) |
| 11 | Report Output | Structured JSON with findings, lesions, grading, recommendations |

### Project Structure

```
chashm-ai/
+-- main.py                      # Application entry point
+-- requirements.txt             # Python dependencies
+-- .env.example                 # Environment configuration template
+-- .gitignore
+-- logo.png                     # Application logo
+-- app/
|   +-- config.py                # Application configuration
|   +-- main_window.py           # Main window with sidebar + stacked views
|   +-- ai/
|   |   +-- engine.py            # AI analysis engine (11-stage pipeline)
|   |   +-- trainer.py           # Model training + synthetic data generation
|   +-- database/
|   |   +-- connection.py        # SQLAlchemy database connection
|   |   +-- models.py            # Doctor, Patient, ScanRecord ORM models
|   |   +-- seed.py              # Default doctor seeding
|   +-- reporting/
|   |   +-- pdf_generator.py     # PDF report generation (ReportLab)
|   +-- services/
|   |   +-- auth_service.py      # Authentication (bcrypt + JWT)
|   |   +-- patient_service.py   # Patient CRUD
|   |   +-- scan_service.py      # Scan management
|   |   +-- report_service.py    # Report management
|   +-- ui/
|   |   +-- components/
|   |   |   +-- sidebar.py       # Navigation sidebar with logo
|   |   |   +-- topbar.py        # Top bar with search + logout
|   |   |   +-- theme.py         # Dark theme stylesheet
|   |   |   +-- stat_card.py     # Dashboard statistics card
|   |   +-- dialogs/
|   |   |   +-- login_dialog.py      # Login/registration
|   |   |   +-- model_loading_dialog.py  # AI model loading animation
|   |   +-- views/
|   |       +-- dashboard_view.py     # Statistics overview
|   |       +-- patient_view.py       # Patient management
|   |       +-- doctor_view.py        # Doctor profile
|   |       +-- scan_view.py          # Scan upload + analysis + results
|   |       +-- reports_view.py       # Report history
|   |       +-- ai_training_view.py   # Model fine-tuning UI
|   |       +-- settings_view.py      # Application settings
|   |       +-- logs_view.py          # System logs
|   +-- utils/
|       +-- helpers.py            # Utility functions
|       +-- validators.py         # Input validation
+-- data/
|   +-- trained_models/           # Trained model weights (.pt files)
|   +-- reports/                  # Generated PDF reports
|   +-- qrcodes/                  # QR code images
|   +-- scans/                    # Uploaded scan images
|   +-- viz/                      # Visualization outputs
+-- tests/
    +-- test_app.py
    +-- test_cv2.py
    +-- test_models.py
```

---

## Supported Diseases

### Fundus (Retinal) Diseases
| Disease | Severity Range | ICD-10 |
|---------|---------------|--------|
| Diabetic Retinopathy | No DR to PDR | E11.3 |
| Glaucoma | Early to End-Stage | H40.9 |
| Cataract | Mild to Very Severe | H26.9 |
| Age-Related Macular Degeneration | Early to Geographic Atrophy | H35.3 |
| Retinal Detachment | - | H33.0 |
| Hypertensive Retinopathy | - | I10.0 |
| Macular Edema | - | H35.8 |
| Retinitis Pigmentosa | - | H35.5 |
| Optic Disc Edema | - | H47.1 |
| Dry Eye Disease | - | H04.1 |

### External Eye Diseases
| Disease | ICD-10 | Key Signs |
|---------|--------|-----------|
| Bacterial Conjunctivitis | H10.0 | Diffuse redness, mucopurulent discharge |
| Viral Conjunctivitis | H10.3 | Watery discharge, follicular reaction |
| Allergic Conjunctivitis | H10.1 | Itching, papillary reaction |
| Subconjunctival Hemorrhage | H11.3 | Bright red patch, well-demarcated |
| Pterygium | H11.0 | Wing-shaped growth, nasal encroachment |
| Pinguecula | H11.1 | Yellow-white nodule |
| Hordeolum (Stye) | H00.0 | Tender red lump, lid margin |
| Chalazion | H00.1 | Non-tender nodule, lid plate |
| Blepharitis | H01.0 | Lid margin redness, scaling |
| Scleritis | H15.0 | Deep violaceous red, severe pain |
| Corneal Ulcer | H16.0 | White opacity, pain, photophobia |
| Scleral Icterus | R17 | Yellow sclera, bilateral |

---

## Installation

### Prerequisites
- Python 3.10+
- pip

### Setup

```bash
# Clone the repository
git clone https://github.com/shhhoaib/Chashm-AI.git
cd Chashm-AI

# Create virtual environment (optional)
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/Mac: source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your settings (optional)

# Run the application
python main.py
```

### Default Login
- **Email**: admin@chashm.ai
- **Password**: admin123

---

## Configuration

### Environment Variables (.env)
| Variable | Default | Description |
|----------|---------|-------------|
| DATABASE_URL | sqlite:///chashm_ai.db | Database connection string |
| SECRET_KEY | chashm-ai-secret-key-... | JWT signing key |
| OPENROUTER_API_KEY | - | OpenRouter API key (for LLM second opinion) |
| OPENROUTER_MODEL | qwen/qwen2.5-vl-72b-instruct:free | Vision LLM model |
| OPENROUTER_BASE_URL | https://openrouter.ai/api/v1 | API base URL |

### OpenRouter Integration (Optional)
To enable the LLM second-opinion stage:
1. Get a free API key at [openrouter.ai/keys](https://openrouter.ai/keys)
2. Set OPENROUTER_API_KEY in your .env file
3. The LLM review activates automatically alongside the CNN ensemble

---

## Usage

### Running the Application
```bash
python main.py
```

### Workflow
1. **Login** with your credentials (default: admin@chashm.ai / admin123)
2. **AI models load** automatically (4 models on CPU/GPU)
3. **Navigate** using the sidebar:
   - **Dashboard** - Overview statistics
   - **Patients** - Manage patient records
   - **Doctors** - View doctor profile
   - **Scan Analysis** - Upload images and run AI analysis
   - **Reports** - View and manage generated reports
   - **AI Training** - Fine-tune models with synthetic data
   - **Settings** - Configure application
   - **Logs** - View system logs
4. **Run Analysis**:
   - Select a patient
   - Upload a fundus or external eye image
   - Click "Run AI Analysis"
   - View results with disease detection, severity, risk, and explanations
   - Generate a professional PDF report

---

## Technical Stack

| Component | Technology |
|-----------|-----------|
| GUI Framework | PySide6 (Qt for Python) |
| AI/ML | PyTorch, torchvision, OpenCV |
| Database | SQLAlchemy + SQLite |
| PDF Generation | ReportLab |
| Auth | bcrypt + PyJWT |
| LLM Integration | OpenRouter API (optional) |
| Image Processing | NumPy, OpenCV, PIL |

---

## Safety & Disclaimer

This application is an AI-assisted screening tool and is NOT a replacement for professional medical diagnosis. All findings are preliminary and require review by a licensed ophthalmologist. The AI system is designed to assist, not replace, clinical judgment.

Key safety features:
- Multi-model ensemble reduces individual model bias
- Safety rules prevent false-negative classifications
- Optional LLM second opinion provides additional verification
- All reports clearly state they are AI-assisted and require specialist review
- Low-confidence and low-quality results are flagged appropriately

---

## License

MIT License - see LICENSE file for details.

## Author

Developed by Shoaib (github.com/shhhoaib)
