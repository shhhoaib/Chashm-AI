import qrcode
from datetime import datetime
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, Image,
)
from reportlab.graphics.shapes import Drawing, Rect, String, Circle
from reportlab.graphics import renderPDF
from reportlab.graphics.barcode.code128 import Code128
from reportlab.platypus.flowables import Flowable

from app.config import APP_NAME, APP_VERSION, REPORTS_DIR, QRCODES_DIR

QR_DIR = QRCODES_DIR
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
QR_DIR.mkdir(parents=True, exist_ok=True)

PRIMARY = "#001F54"
SECONDARY = "#0A84FF"
ACCENT = "#33C3FF"
DARK_TEXT = "#1A1A2E"
BORDER = "#C8CED6"
BG_LIGHT = "#F0F2F5"
BG_CARD = "#FFFFFF"
WHITE = "#FFFFFF"
SUCCESS_GREEN = "#10B981"
WARN_AMBER = "#F59E0B"
DANGER_RED = "#DC2626"
MUTED_TEXT = "#667085"
FOOTER_TEXT = "#98A2B3"

styles = getSampleStyleSheet()

styles.add(ParagraphStyle("ReportTitle", fontName="Helvetica-Bold", fontSize=18,
    textColor=colors.white, spaceAfter=0, alignment=TA_CENTER, leading=22))
styles.add(ParagraphStyle("HospitalName", fontName="Helvetica-Bold", fontSize=12,
    textColor=colors.white, spaceAfter=0, alignment=TA_CENTER, leading=15))
styles.add(ParagraphStyle("ReportTagline", fontName="Helvetica", fontSize=7,
    textColor=colors.HexColor("#CBD5E1"), spaceAfter=0, alignment=TA_CENTER, leading=9))
styles.add(ParagraphStyle("SectionHdr", fontName="Helvetica-Bold", fontSize=9.5,
    textColor=colors.HexColor(PRIMARY), spaceBefore=8, spaceAfter=5,
    borderPadding=6, backColor=colors.HexColor(BG_LIGHT)))
styles.add(ParagraphStyle("InfoHdr", fontName="Helvetica-Bold", fontSize=8,
    textColor=colors.HexColor(PRIMARY), spaceBefore=2, spaceAfter=3))
styles.add(ParagraphStyle("Label", fontName="Helvetica-Bold", fontSize=7.5,
    textColor=colors.HexColor("#344054")))
styles.add(ParagraphStyle("Value", fontName="Helvetica", fontSize=7.5,
    textColor=colors.HexColor(DARK_TEXT), leading=10))
styles.add(ParagraphStyle("FindingsVal", fontName="Helvetica", fontSize=7.5,
    textColor=colors.HexColor(DARK_TEXT), leading=10))
styles.add(ParagraphStyle("RecoVal", fontName="Helvetica", fontSize=7.5,
    textColor=colors.HexColor(DARK_TEXT), leading=10))
styles.add(ParagraphStyle("BulletItem", fontName="Helvetica", fontSize=7,
    textColor=colors.HexColor("#475467"), leading=9, leftIndent=8))
styles.add(ParagraphStyle("Disclaimer", fontName="Helvetica-Oblique", fontSize=6.5,
    textColor=colors.HexColor("#DC2626"), alignment=TA_CENTER, leading=8))
styles.add(ParagraphStyle("Footer", fontName="Helvetica", fontSize=6,
    textColor=colors.HexColor(FOOTER_TEXT), alignment=TA_CENTER, leading=7))


class DrawingFlowable(Flowable):
    def __init__(self, drawing, width, height):
        Flowable.__init__(self)
        self.drawing = drawing
        self.width = width
        self.height = height
        self.h = height

    def draw(self):
        renderPDF.draw(self.drawing, self.canv, 0, 0)


def _make_hospital_logo(size: int = 30) -> DrawingFlowable:
    d = Drawing(size, size)
    bg = colors.HexColor(PRIMARY)
    d.add(Rect(0, 0, size, size, fillColor=bg, strokeColor=None))
    cx, cy = size // 2, size // 2
    cross = colors.white
    w, h = size * 0.55, size * 0.18
    d.add(Rect(cx - w / 2, cy - h / 2, w, h, fillColor=cross, strokeColor=None, radius=2))
    d.add(Rect(cx - h / 2, cy - w / 2, h, w, fillColor=cross, strokeColor=None, radius=2))
    eye_r = size * 0.12
    d.add(Circle(cx, cy, eye_r, fillColor=cross, strokeColor=None))
    d.add(Circle(cx, cy, eye_r * 0.6, fillColor=bg, strokeColor=None))
    return DrawingFlowable(d, size, size)


def _make_severity_badge(severity: str) -> str:
    s = severity.lower()
    if s in ("severe", "high", "proliferative", "advanced", "end-stage", "critical"):
        return f'<table border="0" cellpadding="2"><tr><td bgcolor="{DANGER_RED}" style="border-radius:3px;">&nbsp;<font color="white"><b>{severity}</b></font>&nbsp;</td></tr></table>'
    if s in ("moderate", "medium"):
        return f'<table border="0" cellpadding="2"><tr><td bgcolor="{WARN_AMBER}" style="border-radius:3px;">&nbsp;<font color="white"><b>{severity}</b></font>&nbsp;</td></tr></table>'
    if s in ("mild", "early", "low", "observed", "normal"):
        return f'<table border="0" cellpadding="2"><tr><td bgcolor="{SUCCESS_GREEN}" style="border-radius:3px;">&nbsp;<font color="white"><b>{severity}</b></font>&nbsp;</td></tr></table>'
    return f'<font color="{MUTED_TEXT}">{severity}</font>'


def generate_qr_code(report_number: str) -> str:
    qr = qrcode.QRCode(version=1, error_correction=qrcode.constants.ERROR_CORRECT_L,
                        box_size=5, border=2)
    qr.add_data(f"chashm-ai://verify/{report_number}")
    qr.make(fit=True)
    img = qr.make_image(fill_color=PRIMARY, back_color="white")
    qr_path = QR_DIR / f"{report_number}.png"
    img.save(qr_path)
    return str(qr_path)


DISCLAIMER_TEXT = (
    "This report is AI-assisted and must be reviewed by a licensed ophthalmologist. "
    "This is not a final diagnosis. Always consult a qualified medical professional."
)

DIET_RECOMMENDATIONS = {
    "Diabetic Retinopathy": [
        "Control blood sugar levels with low-glycemic foods",
        "Include leafy greens (spinach, kale) rich in lutein",
        "Eat omega-3 fatty acids (salmon, flaxseeds, walnuts)",
        "Consume antioxidant-rich berries (blueberries, strawberries)",
        "Avoid sugary drinks and refined carbohydrates",
        "Limit saturated fats and processed foods",
    ],
    "Glaucoma": [
        "Include dark leafy greens (kale, collard greens) for nitric oxide",
        "Eat colorful fruits & vegetables for antioxidant protection",
        "Consume omega-3 fatty acids (fish, chia seeds)",
        "Limit caffeine intake to reduce intraocular pressure",
        "Stay hydrated with adequate water throughout the day",
        "Avoid heavy lifting and inverted yoga positions",
    ],
    "Cataract": [
        "Consume vitamin C-rich foods (citrus, bell peppers, broccoli)",
        "Eat vitamin E sources (almonds, sunflower seeds, avocados)",
        "Include lutein & zeaxanthin (eggs, corn, orange peppers)",
        "Eat beta-carotene rich foods (carrots, sweet potatoes)",
        "Limit alcohol consumption and quit smoking",
        "Wear UV-protective sunglasses outdoors",
    ],
    "AMD": [
        "Follow AREDS2 diet: lutein, zeaxanthin, vitamins C & E, zinc",
        "Eat dark leafy greens (spinach, kale, swiss chard) daily",
        "Consume fatty fish (salmon, mackerel, sardines) 2-3x/week",
        "Include eggs (especially yolks for lutein)",
        "Eat orange/yellow vegetables (squash, corn, peppers)",
        "Avoid smoking and protect eyes from blue light",
    ],
    "Hypertensive Retinopathy": [
        "Reduce sodium intake to under 1500mg/day",
        "Follow DASH diet: fruits, vegetables, whole grains, lean protein",
        "Consume potassium-rich foods (bananas, potatoes, beans)",
        "Limit alcohol and avoid tobacco completely",
        "Include beetroot and dark chocolate for nitric oxide",
        "Monitor blood pressure regularly",
    ],
    "Retinal Detachment": [
        "Immediate ophthalmology consultation required",
        "Avoid strenuous activity and heavy lifting",
        "Protect eyes from trauma with safety eyewear",
        "Consume vitamin C & E for retinal health",
        "Include omega-3 fatty acids for inflammation control",
        "Follow all post-surgical dietary restrictions if treated",
    ],
    "Macular Edema": [
        "Reduce sodium intake to manage fluid retention",
        "Control blood sugar if diabetic",
        "Consume anti-inflammatory foods (turmeric, ginger, berries)",
        "Include omega-3 fatty acids (fish oil, flaxseed)",
        "Avoid trans fats and excessive alcohol",
        "Stay hydrated but avoid fluid overload",
    ],
    "Retinitis Pigmentosa": [
        "Take vitamin A palmitate under medical supervision only",
        "Include docosahexaenoic acid (DHA) from fish oil",
        "Consume lutein-rich foods (kale, spinach, egg yolks)",
        "Eat antioxidant-rich foods (berries, nuts, dark chocolate)",
        "Avoid excessive vitamin E supplementation",
        "Use UV-protective eyewear in bright conditions",
    ],
    "Optic Disc Edema": [
        "Immediate neurological and ophthalmological evaluation needed",
        "Reduce sodium intake to decrease fluid pressure",
        "Limit caffeine and stimulant consumption",
        "Avoid strenuous physical activity",
        "Elevate head while sleeping to reduce intracranial pressure",
        "Follow prescribed treatment for underlying cause",
    ],
    "Dry Eye Disease": [
        "Increase omega-3 intake (fish oil, flaxseed supplements)",
        "Stay hydrated: drink 8-10 glasses of water daily",
        "Consume vitamin A-rich foods (carrots, sweet potatoes, liver)",
        "Include healthy fats (avocado, olive oil, nuts)",
        "Avoid dry environments; use humidifier",
        "Take regular screen breaks (20-20-20 rule)",
    ],
    "Conjunctivitis": [
        "Avoid touching or rubbing the eyes",
        "Use separate towels and washcloths to prevent spread",
        "Increase vitamin C intake for immune support",
        "Consume anti-inflammatory foods (turmeric, ginger)",
        "Stay hydrated and get adequate rest",
        "Discard eye makeup and avoid sharing cosmetics",
    ],
    "Subconjunctival Hemorrhage": [
        "Avoid strenuous activity and heavy lifting",
        "Apply cold compresses in the first 24 hours",
        "Avoid blood-thinning medications if possible (consult doctor)",
        "Consume vitamin C and K rich foods for healing",
        "Monitor for recurrence or associated symptoms",
        "Usually resolves spontaneously within 1-2 weeks",
    ],
    "Pterygium": [
        "Wear UV-protective sunglasses outdoors",
        "Use artificial tears for dryness and irritation",
        "Avoid dusty, windy, and dry environments",
        "Include anti-inflammatory foods in diet",
        "Consider omega-3 supplements for inflammation control",
        "Monitor growth with regular eye examinations",
    ],
    "Pinguecula": [
        "Use lubricating eye drops for dryness",
        "Wear UV-protective sunglasses outdoors",
        "Avoid prolonged exposure to dust and wind",
        "Include vitamin A and C rich foods in diet",
        "Use anti-inflammatory eye drops if prescribed",
        "Usually benign; monitor for changes",
    ],
    "Hordeolum": [
        "Apply warm compresses 3-4 times daily",
        "Practice good eyelid hygiene",
        "Avoid squeezing or popping the stye",
        "Clean eyelids with diluted baby shampoo",
        "Increase vitamin C and zinc intake",
        "Replace eye makeup to prevent reinfection",
    ],
    "Chalazion": [
        "Apply warm compresses for 10-15 minutes several times daily",
        "Gently massage the eyelid to promote drainage",
        "Practice good eyelid hygiene",
        "Avoid eye makeup until resolved",
        "Include anti-inflammatory foods in diet",
        "If persistent, seek ophthalmology evaluation",
    ],
    "Blepharitis": [
        "Practice daily eyelid hygiene with warm compresses",
        "Clean eyelid margins with diluted baby shampoo",
        "Use artificial tears for dryness",
        "Include omega-3 fatty acids in diet",
        "Avoid eye makeup during flare-ups",
        "Regular eyelid scrubs as recommended by doctor",
    ],
    "Scleritis": [
        "Seek immediate ophthalmology evaluation",
        "Take prescribed anti-inflammatory medications",
        "Apply cold compresses for comfort",
        "Avoid strenuous activity",
        "Include anti-inflammatory foods (turmeric, ginger, berries)",
        "Monitor for associated systemic conditions",
    ],
    "Corneal Ulcer": [
        "Seek immediate ophthalmology emergency care",
        "Do not wear contact lenses until healed",
        "Avoid touching or rubbing the eye",
        "Use prescribed antibiotic/antifungal eye drops",
        "Follow strict hand hygiene",
        "Attend all follow-up appointments",
    ],
}

PATIENT_EXPLANATIONS = {
    "Diabetic Retinopathy": (
        "This condition is caused by high blood sugar damaging the tiny blood vessels in the back of your eye (retina). "
        "In early stages, it may not cause any symptoms, but if it progresses, it can lead to vision loss. "
        "The good news is that controlling your blood sugar, blood pressure, and cholesterol can slow or stop the damage."
    ),
    "Glaucoma": (
        "Glaucoma is a condition where the optic nerve — the cable that connects your eye to your brain — becomes damaged, "
        "usually due to higher-than-normal pressure inside the eye. It often has no early symptoms, which is why regular "
        "check-ups matter. Treatment with eye drops or surgery can prevent further vision loss."
    ),
    "Cataract": (
        "A cataract is when the natural lens inside your eye becomes cloudy, like a foggy window. This makes vision blurry, "
        "especially at night or in bright light. Cataracts develop slowly with age and are very common. "
        "Surgery to replace the cloudy lens with a clear artificial one is safe and highly effective."
    ),
    "AMD": (
        "Age-related Macular Degeneration (AMD) affects the central part of your retina (the macula) that you use for "
        "reading, recognizing faces, and seeing fine details. It does not cause complete blindness because side vision "
        "remains. A special diet (AREDS2) and certain treatments can slow its progression."
    ),
    "Hypertensive Retinopathy": (
        "High blood pressure can damage the small blood vessels in your retina. This is a sign that your blood pressure "
        "may need better control. Managing your blood pressure through diet, exercise, and medication can prevent "
        "further damage to both your eyes and your overall health."
    ),
    "Retinal Detachment": (
        "This is a medical emergency where the retina peels away from the back of the eye, like wallpaper peeling off "
        "a wall. It causes sudden flashes, floaters, or a curtain-like shadow over your vision. "
        "Immediate surgery is needed to prevent permanent vision loss."
    ),
    "Macular Edema": (
        "This is swelling in the central part of the retina (macula) caused by fluid leaking from damaged blood vessels. "
        "It often occurs in people with diabetes or after eye surgery. Treatment with injections or lasers can "
        "reduce the swelling and help preserve vision."
    ),
    "Retinitis Pigmentosa": (
        "This is a genetic condition that causes the light-sensitive cells in your retina to gradually break down. "
        "It usually starts with difficulty seeing at night (night blindness) and slowly progresses to tunnel vision. "
        "While there is no cure, vitamin A therapy and low-vision aids can help."
    ),
    "Optic Disc Edema": (
        "This is swelling of the optic disc where the optic nerve enters your eye. It can be caused by increased "
        "pressure inside the skull, high blood pressure, or inflammation. Finding and treating the underlying cause "
        "is essential to protect your vision."
    ),
    "Dry Eye Disease": (
        "Dry eye means your eyes do not produce enough tears or the tears evaporate too quickly. This can make "
        "your eyes feel scratchy, burning, or tired, especially during screen use. Artificial tears, warm compresses, "
        "and omega-3 supplements can provide relief."
    ),
    "Conjunctivitis": (
        "Also called 'pink eye,' this is inflammation of the thin clear tissue covering the white part of your eye. "
        "It can be caused by bacteria, viruses, or allergies. Bacterial conjunctivitis needs antibiotic drops; "
        "viral and allergic types usually improve on their own or with antihistamine drops."
    ),
    "Subconjunctival Hemorrhage": (
        "This looks scary — a bright red patch on the white of your eye — but it is usually harmless. It happens "
        "when a tiny blood vessel bursts, often from coughing, sneezing, or straining. It clears up on its own "
        "within 1 to 2 weeks, like a bruise on your eye."
    ),
    "Pterygium": (
        "This is a fleshy, triangular growth of tissue that extends from the white of the eye toward the cornea. "
        "It is often caused by long-term exposure to UV light, dust, or wind. If it grows large enough to affect "
        "vision, it can be removed surgically."
    ),
    "Pinguecula": (
        "This is a small yellowish bump on the white of the eye, usually on the side closest to your nose. It is "
        "caused by UV exposure, dust, or dry eyes. It is benign and rarely needs treatment beyond lubricating "
        "eye drops. Wear sunglasses outdoors to prevent it."
    ),
    "Hordeolum": (
        "Commonly called a 'stye,' this is a red, painful lump on the edge of your eyelid caused by a bacterial "
        "infection in an oil gland. Warm compresses applied several times a day can help it drain and heal. "
        "Do not squeeze it — that can spread the infection."
    ),
    "Chalazion": (
        "This is a firm, painless lump on the eyelid that forms when an oil gland gets blocked. Unlike a stye, "
        "it is not an infection. Warm compresses and gentle massage can help it go away. If it persists for "
        "months, a doctor can drain it."
    ),
    "Blepharitis": (
        "This is inflammation of the eyelid margins, causing redness, flaking, and irritation. It is often "
        "linked to dandruff or rosacea. Daily eyelid hygiene — warm compresses, gentle scrubs — is the main "
        "treatment. It is a chronic condition that needs ongoing care."
    ),
    "Scleritis": (
        "This is a deep, painful inflammation of the white outer coating of the eye (sclera). It is more serious "
        "than conjunctivitis and is often linked to autoimmune conditions like rheumatoid arthritis. "
        "It needs urgent evaluation by an ophthalmologist and treatment with anti-inflammatory medications."
    ),
    "Corneal Ulcer": (
        "This is an open sore on the clear front surface of your eye (cornea), often caused by an infection. "
        "It is extremely serious and can lead to vision loss or even perforation of the eye if not treated "
        "promptly. This is a medical emergency — seek immediate eye care."
    ),
    "Scleral Icterus": (
        "This is a yellow discoloration of the whites of the eyes. It is usually a sign of jaundice, which "
        "indicates a problem with the liver or gallbladder. While not an eye disease itself, it requires "
        "medical evaluation to address the underlying cause."
    ),
    "Keratitis": (
        "This is inflammation of the cornea, often caused by infection (especially in contact lens wearers) or injury. "
        "It causes pain, redness, blurred vision, and sensitivity to light. Prompt treatment with medicated "
        "eye drops is essential to prevent scarring and vision loss."
    ),
    "Keratoconus": (
        "This is a condition where the cornea gradually thins and bulges into a cone shape, causing blurry and "
        "distorted vision. It usually starts in the teenage years. Special contact lenses can help, and in "
        "advanced cases, corneal transplant surgery may be needed."
    ),
}

LIFESTYLE_RECOMMENDATIONS = [
    "Schedule regular comprehensive eye examinations",
    "Use appropriate UV protection outdoors",
    "Maintain healthy blood pressure and cholesterol levels",
    "Exercise regularly (30 min moderate activity, 5 days/week)",
    "Maintain healthy body weight (BMI 18.5-24.9)",
    "Avoid smoking and limit alcohol consumption",
    "Manage screen time with regular breaks (20-20-20 rule)",
    "Follow a balanced Mediterranean-style diet",
]

SEVERITY_ACTIONS = {
    "Critical": "URGENT: Immediate ophthalmology referral within 24 hours.",
    "High": "Urgent ophthalmology referral within 24-48 hours.",
    "Moderate": "Schedule follow-up with ophthalmologist within 1-2 weeks.",
    "Medium": "Schedule follow-up with ophthalmologist within 1-2 weeks.",
    "Low": "Routine follow-up in 3-6 months or per regular schedule.",
    "Normal": "Routine follow-up as per regular schedule.",
}


def _get_patient_explanation(disease_name: str) -> str:
    if not disease_name:
        return ""
    for key, explanation in PATIENT_EXPLANATIONS.items():
        if key.lower() in disease_name.lower() or disease_name.lower() in key.lower():
            return explanation
    return ""


def _get_dietary_advice(disease_name: str) -> list:
    if not disease_name:
        return [
            "Maintain a balanced Mediterranean-style diet",
            "Include antioxidant-rich fruits and vegetables daily",
            "Consume omega-3 fatty acids from fish or plant sources",
            "Stay hydrated with adequate water intake",
        ]
    for key, recs in DIET_RECOMMENDATIONS.items():
        if key.lower() in disease_name.lower() or disease_name.lower() in key.lower():
            return recs
    return [
        "Maintain a balanced Mediterranean-style diet",
        "Include antioxidant-rich fruits and vegetables daily",
        "Consume omega-3 fatty acids from fish or plant sources",
        "Stay hydrated with adequate water intake",
    ]


def _status_color(severity: str) -> colors.Color:
    s = severity.lower()
    if s in ("severe", "high", "proliferative", "advanced", "end-stage", "critical"):
        return colors.HexColor(DANGER_RED)
    if s in ("moderate", "medium"):
        return colors.HexColor(WARN_AMBER)
    if s in ("mild", "early", "low", "observed", "normal"):
        return colors.HexColor(SUCCESS_GREEN)
    return colors.HexColor(MUTED_TEXT)


def _make_barcode(report_number: str, width=140, height=30) -> DrawingFlowable:
    try:
        barcode = Code128(report_number, barHeight=height * 0.7, barWidth=0.8)
        barcode.drawWidth = width
        barcode.drawHeight = height
        d = Drawing(width, height + 8)
        d.add(barcode)
        d.add(String(width / 2, 0, report_number, fontName="Helvetica", fontSize=6,
                      textAnchor="middle", fillColor=colors.HexColor(MUTED_TEXT)))
        return DrawingFlowable(d, width, height + 8)
    except Exception:
        d = Drawing(width, height)
        d.add(String(width / 2, height / 2, report_number,
                      fontName="Helvetica", fontSize=7, textAnchor="middle"))
        return DrawingFlowable(d, width, height)


def _info_row(label: str, value: str) -> list:
    return [Paragraph(f"<b>{label}:</b>", styles["Label"]),
            Paragraph(str(value) if value else "N/A", styles["Value"])]


def _safe_score(value, default=0):
    if value is None:
        return default
    if isinstance(value, str):
        try:
            return float(value.replace("%", "").strip())
        except (ValueError, TypeError):
            return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def generate_report(data: dict) -> str:
    report_number = data.get("report_number", "N/A")
    pdf_filename = f"{report_number}.pdf"
    pdf_path = str(REPORTS_DIR / pdf_filename)

    doc = SimpleDocTemplate(pdf_path, pagesize=A4,
        topMargin=12, bottomMargin=14, leftMargin=18, rightMargin=18)
    elements = []
    W = A4[0] - 36

    now = datetime.now()
    now_date_str = now.strftime("%d %B %Y")
    now_time_str = now.strftime("%I:%M %p")
    now_full_str = f"{now_date_str} at {now_time_str}"

    doctor = data.get("doctor", {})
    patient = data.get("patient", {})
    scan = data.get("scan", {})
    disease = scan.get("disease") or {}
    findings_list = scan.get("findings", [])
    recommendation_text = scan.get("recommendation", "")
    disclaimer = scan.get("disclaimer", "")
    image_type = scan.get("image_type", data.get("image_type", "fundus"))
    risk_level = scan.get("risk_level", disease.get("risk", "Low"))
    severity_level = scan.get("severity_level", disease.get("severity", "Normal"))
    disease_name = disease.get("disease", "") if disease else ""
    severity_color = _status_color(severity_level)
    dietary = _get_dietary_advice(disease_name)
    follow_up = data.get("follow_up", "")
    clinical_summary = data.get("clinical_summary", "")

    hospital_name = doctor.get("hospital_name", "Hospital")
    doctor_name = doctor.get("full_name", "Attending Physician")
    doctor_qual = doctor.get("qualification", "N/A")
    doctor_spec = doctor.get("specialization", "N/A")
    doctor_reg = doctor.get("registration_number", "N/A")
    doctor_mobile = doctor.get("mobile_number", "N/A")

    # ── HEADER ──────────────────────────────────────────────────
    logo_flowable = _make_hospital_logo(32)

    header_content = [
        [Paragraph(f"<b>{hospital_name}</b>", styles["HospitalName"])],
        [Paragraph(APP_NAME + " \u2014 AI-Assisted Ophthalmic Screening Report", styles["ReportTagline"])],
    ]
    header_table_inner = Table(header_content, colWidths=[W - 52])
    header_table_inner.setStyle(TableStyle([
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
    ]))

    header_outer = Table([[logo_flowable, header_table_inner]], colWidths=[42, W - 42])
    header_outer.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(PRIMARY)),
        ("TEXTCOLOR", (0, 0), (-1, -1), colors.white),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    elements.append(header_outer)

    header_meta = Table([
        [Paragraph(f"<b>Report #{report_number}</b>", ParagraphStyle("hm",
            fontName="Helvetica-Bold", fontSize=7, textColor=colors.HexColor("#CBD5E1"), alignment=TA_LEFT)),
         Paragraph(f"<b>Issued:</b> {now_full_str}", ParagraphStyle("hm2",
            fontName="Helvetica", fontSize=7, textColor=colors.HexColor("#CBD5E1"), alignment=TA_RIGHT))]
    ], colWidths=[W * 0.5, W * 0.5])
    header_meta.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(PRIMARY)),
        ("LEFTPADDING", (0, 0), (0, 0), 10),
        ("RIGHTPADDING", (0, 0), (1, 0), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 1),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    elements.append(header_meta)

    # ── PATIENT & DOCTOR INFORMATION ────────────────────────────
    elements.append(Paragraph("PATIENT &amp; DOCTOR INFORMATION", styles["SectionHdr"]))

    patient_name = patient.get("name", "N/A")
    patient_id = patient.get("patient_id", "N/A")
    patient_age = patient.get("age", "N/A")
    patient_gender = patient.get("gender", "N/A")
    patient_mobile = patient.get("mobile_number", "N/A")

    info_rows = [
        [Paragraph("<b>PATIENT INFORMATION</b>", styles["InfoHdr"]),
         Paragraph("<b>DOCTOR INFORMATION</b>", styles["InfoHdr"])],
        _info_row("Name", patient_name) + _info_row("Name", doctor_name),
        _info_row("Patient ID", patient_id) + _info_row("Registration No.", doctor_reg),
        _info_row("Age / Gender", f"{patient_age} / {patient_gender}") + _info_row("Qualification", doctor_qual),
        _info_row("Mobile", patient_mobile) + _info_row("Specialization", doctor_spec),
    ]

    info_table = Table(info_rows, colWidths=[W * 0.28, W * 0.22, W * 0.28, W * 0.22])
    info_table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor(BORDER)),
        ("INNERGRID", (0, 0), (-1, -1), 0.3, colors.HexColor(BORDER)),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(BG_LIGHT)),
    ]))
    elements.append(info_table)

    # ── CLINICAL SUMMARY ────────────────────────────────────────
    if clinical_summary:
        elements.append(Paragraph("CLINICAL SUMMARY", styles["SectionHdr"]))
        elements.append(Paragraph(clinical_summary, styles["Value"]))

    # ── DIAGNOSTIC ANALYSIS ─────────────────────────────────────
    elements.append(Paragraph("DIAGNOSTIC ANALYSIS REPORT", styles["SectionHdr"]))

    disease_label = disease_name if disease_name else ("No disease detected" if image_type == "fundus" else "N/A")
    severity_badge = _make_severity_badge(severity_level)
    risk_badge = _make_severity_badge(risk_level)
    disease_display = f"<font color='{severity_color.hexval()}'><b>{disease_label}</b></font>"

    conf_score = _safe_score(scan.get("confidence_score", 0))
    if conf_score > 0 and conf_score < 45:
        disease_display += '&nbsp;<table border="0" cellpadding="1"><tr><td bgcolor="#FEF3C7" style="border-radius:2px;">&nbsp;<font color="#92400E" size="5">Inconclusive</font>&nbsp;</td></tr></table>'

    icd_code = ""
    for f_item in findings_list:
        icd = f_item.get("icd_10_code", "")
        if icd:
            icd_code = icd
            break
    if not icd_code and disease:
        icd_code = disease.get("icd_10_code", "")

    quality_score_val = _safe_score(scan.get("quality", {}).get("score", 0))

    res_data = [
        [Paragraph("<b>Parameter</b>", styles["Label"]),
         Paragraph("<b>Result</b>", styles["Label"])],
        [Paragraph("Image Type", styles["Label"]),
         Paragraph(image_type.replace("_", " ").title(), styles["Value"])],
        [Paragraph("Quality Score", styles["Label"]),
         Paragraph(f'{quality_score_val:.1f}%', styles["Value"])],
        [Paragraph("Primary Finding", styles["Label"]),
         Paragraph(disease_display, styles["Value"])],
        [Paragraph("ICD-10 Code", styles["Label"]),
         Paragraph(f"<b>{icd_code}</b>" if icd_code else "<i>Not assigned</i>", styles["Value"])],
        [Paragraph("Confidence", styles["Label"]),
         Paragraph(f'{conf_score:.1f}%' if conf_score > 0 else "<i>N/A</i>", styles["Value"])],
        [Paragraph("Severity", styles["Label"]),
         Paragraph(severity_badge, styles["Value"])],
        [Paragraph("Risk Level", styles["Label"]),
         Paragraph(risk_badge, styles["Value"])],
    ]

    style_cmds = [
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 2.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor(BORDER)),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [WHITE, colors.HexColor(BG_LIGHT)]),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(PRIMARY)),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ]
    res_table = Table(res_data, colWidths=[W * 0.22, W * 0.78], repeatRows=1)
    res_table.setStyle(TableStyle(style_cmds))
    elements.append(res_table)

    # ── CLINICAL FINDINGS ───────────────────────────────────────
    if findings_list and image_type in ("fundus", "external_eye"):
        elements.append(Paragraph("CLINICAL FINDINGS", styles["SectionHdr"]))
        find_data = [[Paragraph("<b>Observation</b>", styles["Label"]),
                      Paragraph("<b>ICD-10</b>", styles["Label"]),
                      Paragraph("<b>Confidence</b>", styles["Label"]),
                      Paragraph("<b>Status</b>", styles["Label"])]]
        for f in findings_list:
            obs = f.get("observation", "")
            conf_val = f.get("confidence", "N/A")
            sev = f.get("severity", "")
            c = _status_color(sev)
            icd = f.get("icd_10_code", "")
            find_data.append([
                Paragraph(f"<font color='{c.hexval()}'>\u25cf</font>&nbsp;{obs}", styles["Value"]),
                Paragraph(f"<b>{icd}</b>" if icd else "-", styles["Value"]),
                Paragraph(str(conf_val), styles["Value"]),
                Paragraph(_make_severity_badge(sev), styles["Value"]),
            ])
        f_style_cmds = [
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 2.5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor(BORDER)),
            ("ROWBACKGROUNDS", (0, 0), (-1, -1), [WHITE, colors.HexColor(BG_LIGHT)]),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(PRIMARY)),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ]
        f_table = Table(find_data, colWidths=[W * 0.40, W * 0.16, W * 0.16, W * 0.28], repeatRows=1)
        f_table.setStyle(TableStyle(f_style_cmds))
        elements.append(f_table)

    # ── WHAT THIS MEANS (PATIENT EXPLANATION) ────────────────────
    explanation = _get_patient_explanation(disease_name)
    if explanation:
        elements.append(Paragraph("WHAT THIS MEANS", styles["SectionHdr"]))
        elements.append(Paragraph(
            f'<font color="#344054">{explanation}</font>',
            styles["Value"]
        ))

    # ── RECOMMENDATIONS & CARE PLAN ─────────────────────────────
    elements.append(Paragraph("RECOMMENDATIONS &amp; CARE PLAN", styles["SectionHdr"]))

    rec_blocks = []
    rec_risk = risk_level if risk_level != "Medium" else "Moderate"
    if rec_risk not in SEVERITY_ACTIONS:
        rec_risk = "Low"
    action = SEVERITY_ACTIONS.get(rec_risk, "Routine follow-up per regular schedule.")
    action_color = DANGER_RED if rec_risk in ("Critical", "High") else WARN_AMBER if rec_risk in ("Moderate", "Medium") else SUCCESS_GREEN
    rec_blocks.append(
        f'<table border="0" cellpadding="3"><tr><td bgcolor="{action_color}" style="border-radius:3px;">'
        f'&nbsp;<font color="white"><b>CLINICAL ACTION:</b> {action}</font>&nbsp;</td></tr></table>'
    )

    if follow_up:
        rec_blocks.append(f"<b>Follow-Up:</b> {follow_up}")

    if recommendation_text:
        rec_blocks.append(f"<b>Doctor Note:</b> {recommendation_text}")

    if dietary:
        rec_blocks.append("<b>Dietary Recommendations:</b>")
        for r in dietary[:5]:
            rec_blocks.append(f"\u2022 {r}")

    rec_blocks.append("<b>Lifestyle Advice:</b>")
    for r in LIFESTYLE_RECOMMENDATIONS[:6]:
        rec_blocks.append(f"\u2022 {r}")

    elements.append(Paragraph("<br/>".join(rec_blocks), styles["RecoVal"]))

    # ── EYE SCAN IMAGE ──────────────────────────────────────────
    annotated_path = scan.get("annotated_path") or scan.get("image_path")
    if annotated_path:
        try:
            import os as _os
            if _os.path.exists(annotated_path):
                eye_img = Image(annotated_path, width=W * 0.40, height=W * 0.40 * 0.75)
                img_table = Table([[eye_img]], colWidths=[W])
                img_table.setStyle(TableStyle([
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("TOPPADDING", (0, 0), (-1, -1), 6),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
                ]))
                elements.append(img_table)
        except Exception:
            pass

    # ── BARCODE + QR + SIGNATURE ────────────────────────────────
    elements.append(Spacer(1, 8))
    elements.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor(BORDER)))

    qr_path = data.get("qr_code_path") or generate_qr_code(report_number)
    try:
        qr_img = Image(qr_path, width=38, height=38)
    except Exception:
        qr_img = Paragraph("QR: N/A", ParagraphStyle("sq", fontSize=6))

    barcode_flowable = _make_barcode(report_number, width=150, height=28)

    footer_row = [
        barcode_flowable,
        qr_img,
        Paragraph(
            f"<b>Verify at:</b> chashm-ai://verify/{report_number}<br/>"
            f"<font size=5>Report #{report_number}</font>",
            ParagraphStyle("vfy", fontName="Helvetica", fontSize=6, alignment=TA_CENTER, leading=8)),
        Paragraph(
            "_________________________<br/><font size=6>Doctor Signature</font>",
            ParagraphStyle("sig1", fontName="Helvetica", fontSize=6.5, alignment=TA_CENTER, leading=9)),
        Paragraph(
            "_________________________<br/><font size=6>Hospital Stamp</font>",
            ParagraphStyle("sig2", fontName="Helvetica", fontSize=6.5, alignment=TA_CENTER, leading=9)),
    ]
    footer_table = Table([footer_row], colWidths=[140, 42, W * 0.24, W * 0.22, W * 0.20])
    footer_table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (0, 0), (0, 0), "CENTER"),
        ("ALIGN", (1, 0), (1, 0), "CENTER"),
        ("LEFTPADDING", (0, 0), (-1, -1), 2),
        ("RIGHTPADDING", (0, 0), (-1, -1), 2),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    elements.append(footer_table)

    # ── DISCLAIMER & FOOTER ─────────────────────────────────────
    elements.append(Spacer(1, 3))
    elements.append(HRFlowable(width="100%", thickness=0.3, color=colors.HexColor(BORDER)))
    disclaim = disclaimer if disclaimer else DISCLAIMER_TEXT
    elements.append(Paragraph(disclaim, styles["Disclaimer"]))
    elements.append(Paragraph(
        f"{APP_NAME} v{APP_VERSION} | Report #{report_number} | Generated {now_full_str} | "
        f"Reviewed by: {doctor_name}",
        styles["Footer"]))
    elements.append(Paragraph(
        f"\u00a9 {now.year} {APP_NAME}. All rights reserved. "
        f"Developed by {APP_NAME} Technologies.",
        ParagraphStyle("Copyright", fontName="Helvetica", fontSize=5.5,
            textColor=colors.HexColor(FOOTER_TEXT), alignment=TA_CENTER, leading=7, spaceBefore=2)))

    doc.build(elements)
    return pdf_path
