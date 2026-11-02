import streamlit as st
import pandas as pd
from PIL import Image
from ultralytics import YOLO
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from reportlab.lib import colors
from datetime import datetime
import io
import os
import uuid
import csv
import sqlite3
import hashlib
import secrets
import hmac

# =========================================================
# PROJECT CONFIGURATION
# =========================================================

PROJECT_NAME = (
    "Civix AI – AI-Powered Road Damage Detection "
    "and Complaint Management System"
)

st.set_page_config(
    page_title="Civix AI",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = r"D:\RoadDamageAI"

DATA_DIR = os.path.join(BASE_DIR, "data")
REPORT_DIR = os.path.join(BASE_DIR, "reports")
IMAGE_DIR = os.path.join(REPORT_DIR, "images")
COMPLAINT_IMAGE_DIR = os.path.join(REPORT_DIR, "complaints")

DB_FILE = os.path.join(DATA_DIR, "road_ai.db")

HISTORY_FILE = os.path.join(
    REPORT_DIR, "detection_history.csv"
)

MODEL_PATH = os.path.join(
    BASE_DIR, "runs", "road_damage_v1",
    "weights", "best.pt"
)

for folder in [
    DATA_DIR,
    REPORT_DIR,
    IMAGE_DIR,
    COMPLAINT_IMAGE_DIR
]:
    os.makedirs(folder, exist_ok=True)

HISTORY_COLUMNS = [
    "Report ID", "Date", "Image", "Damage Type",
    "Confidence", "Original Image Path",
    "Annotated Image Path"
]

DAMAGE_TYPES = [
    "pothole",
    "cracks",
    "open_manhole"
]

DAMAGE_LABELS = {
    "pothole": "Pothole",
    "cracks": "Cracks",
    "open_manhole": "Open Manhole"
}

COMPLAINT_STATUSES = [
    "Pending",
    "In Progress",
    "Resolved"
]

# =========================================================
# DATABASE
# =========================================================

def get_db():
    connection = sqlite3.connect(DB_FILE)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_database():
    connection = get_db()
    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            salt TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'user',
            created_at TEXT NOT NULL
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS complaints (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            complaint_id TEXT UNIQUE NOT NULL,
            user_id INTEGER NOT NULL,
            name TEXT NOT NULL,
            address TEXT NOT NULL,
            phone TEXT NOT NULL,
            damage_type TEXT NOT NULL,
            location TEXT NOT NULL,
            description TEXT NOT NULL,
            image_path TEXT,
            status TEXT NOT NULL DEFAULT 'Pending',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    """)

    connection.commit()
    connection.close()


initialize_database()

# =========================================================
# PASSWORD SECURITY
# =========================================================

def hash_password(password, salt=None):
    if salt is None:
        salt = secrets.token_hex(16)

    password_hash = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        300000
    ).hex()

    return password_hash, salt


def verify_password(password, saved_hash, salt):
    calculated_hash, _ = hash_password(
        password, salt
    )

    return hmac.compare_digest(
        calculated_hash,
        saved_hash
    )


def create_user(name, username, password, role="user"):
    password_hash, salt = hash_password(password)

    connection = get_db()

    try:
        connection.execute("""
            INSERT INTO users
            (name, username, password_hash, salt,
             role, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            name.strip(),
            username.strip().lower(),
            password_hash,
            salt,
            role,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ))

        connection.commit()
        return True, "Account created successfully."

    except sqlite3.IntegrityError:
        return False, "This username is already registered."

    finally:
        connection.close()


def authenticate_user(username, password):
    connection = get_db()

    user = connection.execute("""
        SELECT * FROM users WHERE username = ?
    """, (username.strip().lower(),)).fetchone()

    connection.close()

    if user is None:
        return None

    if verify_password(
        password,
        user["password_hash"],
        user["salt"]
    ):
        return dict(user)

    return None


def admin_exists():
    connection = get_db()

    result = connection.execute("""
        SELECT COUNT(*) AS total
        FROM users
        WHERE role = 'admin'
    """).fetchone()

    connection.close()

    return result["total"] > 0


# =========================================================
# COMPLAINT MANAGEMENT
# =========================================================

def create_complaint(
    user_id,
    name,
    address,
    phone,
    damage_type,
    location,
    description,
    image_path
):
    now = datetime.now()

    complaint_id = (
        "CMP-"
        + now.strftime("%Y%m%d")
        + "-"
        + uuid.uuid4().hex[:7].upper()
    )

    connection = get_db()

    connection.execute("""
        INSERT INTO complaints (
            complaint_id,
            user_id,
            name,
            address,
            phone,
            damage_type,
            location,
            description,
            image_path,
            status,
            created_at,
            updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        complaint_id,
        user_id,
        name.strip(),
        address.strip(),
        phone.strip(),
        damage_type,
        location.strip(),
        description.strip(),
        image_path,
        "Pending",
        now.strftime("%Y-%m-%d %H:%M:%S"),
        now.strftime("%Y-%m-%d %H:%M:%S")
    ))

    connection.commit()
    connection.close()

    return complaint_id


def get_user_complaints(user_id):
    connection = get_db()

    rows = connection.execute("""
        SELECT complaint_id, damage_type, location,
               status, created_at, updated_at
        FROM complaints
        WHERE user_id = ?
        ORDER BY id DESC
    """, (user_id,)).fetchall()

    connection.close()

    return [dict(row) for row in rows]


def get_complaint_by_id(complaint_id):
    connection = get_db()

    row = connection.execute("""
        SELECT complaint_id, damage_type, status,
               created_at, updated_at
        FROM complaints
        WHERE complaint_id = ?
    """, (complaint_id.strip().upper(),)).fetchone()

    connection.close()

    return dict(row) if row else None


def get_all_complaints():
    connection = get_db()

    rows = connection.execute("""
        SELECT c.*, u.username
        FROM complaints c
        JOIN users u ON c.user_id = u.id
        ORDER BY c.id DESC
    """).fetchall()

    connection.close()

    return [dict(row) for row in rows]


def update_complaint_status(complaint_id, status):
    connection = get_db()

    connection.execute("""
        UPDATE complaints
        SET status = ?, updated_at = ?
        WHERE complaint_id = ?
    """, (
        status,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        complaint_id
    ))

    connection.commit()
    connection.close()


# =========================================================
# DAMAGE EXPLANATION AND SOLUTIONS
# =========================================================

DAMAGE_GUIDANCE = {
    "pothole": {
        "title": "Pothole Detected",
        "problem": (
            "A pothole is a depression or hole in the road surface. "
            "It may affect vehicle movement and road safety."
        ),
        "solution": (
            "The damaged road area should be inspected by the "
            "concerned road maintenance authority. The damaged "
            "surface may require suitable patching and resurfacing."
        ),
        "priority": "High"
    },

    "cracks": {
        "title": "Road Cracks Detected",
        "problem": (
            "Cracks are visible breaks or openings in the road "
            "surface. If left unattended, they may become wider "
            "and allow water to enter the pavement."
        ),
        "solution": (
            "The road surface should be inspected to identify "
            "the crack severity. Suitable crack sealing or road "
            "surface repair may be required."
        ),
        "priority": "Medium"
    },

    "open_manhole": {
        "title": "Open Manhole Detected",
        "problem": (
            "An open or uncovered manhole can create a serious "
            "road safety risk for pedestrians and vehicles."
        ),
        "solution": (
            "The location should be inspected promptly by the "
            "responsible authority. The opening should be secured "
            "and a suitable manhole cover should be installed."
        ),
        "priority": "Urgent"
    }
}


def show_damage_guidance(detections):
    st.markdown("### Problem Explanation & Suggested Solution")

    if not detections:
        st.info(
            "The AI model did not detect any supported road damage "
            "in this image. This does not guarantee that the road "
            "is completely damage-free."
        )
        return

    shown_types = set()

    for detection in detections:
        damage_type = detection["class"]

        if damage_type in shown_types:
            continue

        shown_types.add(damage_type)

        guidance = DAMAGE_GUIDANCE.get(damage_type)

        if not guidance:
            continue

        st.markdown(f"#### {guidance['title']}")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<div class="panel">', unsafe_allow_html=True)
            st.markdown("**Problem Identified**")
            st.write(guidance["problem"])
            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="panel">', unsafe_allow_html=True)
            st.markdown("**Suggested Solution**")
            st.write(guidance["solution"])
            st.markdown("</div>", unsafe_allow_html=True)

        priority = guidance["priority"]

        if priority == "Urgent":
            st.error(f"Priority: {priority}")
        elif priority == "High":
            st.warning(f"Priority: {priority}")
        else:
            st.info(f"Priority: {priority}")

    st.caption(
        "These are general demonstration suggestions. "
        "A qualified road inspection professional should verify "
        "the actual condition."
    )


# =========================================================
# PROFESSIONAL UI DESIGN - IMPROVED READABILITY
# =========================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');

:root {
    --background: #101820;
    --panel: #1b2935;
    --border: #405665;
    --text: #f1f6fa;
    --muted: #c0ced8;
    --teal: #20c6b0;
}

.stApp {
    background: radial-gradient(
        circle at 85% 0%,
        #1b3440 0%,
        #101820 45%
    );
    color: var(--text);
    font-family: 'DM Sans', sans-serif;
}

[data-testid="stHeader"] {
    background: rgba(16, 24, 32, 0.95);
}

[data-testid="stSidebar"] {
    background: #141f29;
    border-right: 1px solid #344957;
}

[data-testid="stSidebar"] > div {
    padding-top: 1.5rem;
}

[data-testid="stSidebar"] * {
    color: #edf4f8;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1450px;
}

h1, h2, h3, h4, h5, h6 {
    color: #f4f8fb !important;
    font-family: 'Manrope', sans-serif !important;
}

p, label, li {
    color: #d4e0e8;
}

.stMarkdown,
.stText,
[data-testid="stCaptionContainer"] {
    color: #d4e0e8;
}

small {
    color: #c0ced8 !important;
}

.stCaption {
    color: #c0ced8 !important;
}

[data-testid="stWidgetLabel"] p,
[data-testid="stWidgetLabel"] label,
.stTextInput label,
.stTextArea label,
.stSelectbox label,
.stNumberInput label,
.stMultiSelect label,
.stDateInput label,
.stTimeInput label,
.stRadio label,
.stCheckbox label,
.stFileUploader label {
    color: #f0f5f8 !important;
    font-weight: 600 !important;
}

.stTextInput input,
.stTextArea textarea,
.stNumberInput input,
.stDateInput input,
.stTimeInput input {
    background-color: #223442 !important;
    color: #ffffff !important;
    border: 1px solid #607786 !important;
    border-radius: 9px !important;
    caret-color: #20c6b0 !important;
}

.stTextInput input::placeholder,
.stTextArea textarea::placeholder,
.stNumberInput input::placeholder {
    color: #b4c3ce !important;
    opacity: 1 !important;
}

.stTextInput input:focus,
.stTextArea textarea:focus,
.stNumberInput input:focus,
.stDateInput input:focus,
.stTimeInput input:focus {
    border-color: #20c6b0 !important;
    box-shadow: 0 0 0 2px rgba(32, 198, 176, 0.18) !important;
}

.stSelectbox div[data-baseweb="select"] > div,
.stMultiSelect div[data-baseweb="select"] > div {
    background-color: #223442 !important;
    color: #ffffff !important;
    border-color: #607786 !important;
}

.stSelectbox div[data-baseweb="select"] span,
.stMultiSelect div[data-baseweb="select"] span {
    color: #ffffff !important;
}

[data-baseweb="popover"],
[data-baseweb="menu"],
[data-baseweb="list"] {
    background-color: #223442 !important;
}

[data-baseweb="menu"] *,
[data-baseweb="list"] * {
    color: #ffffff !important;
}

[role="option"] {
    background-color: #223442 !important;
    color: #ffffff !important;
}

[role="option"]:hover {
    background-color: #315061 !important;
}

.stRadio [role="radiogroup"] label,
.stCheckbox label {
    color: #eaf2f7 !important;
}

.stRadio [role="radiogroup"] p {
    color: #eaf2f7 !important;
}

.stCheckbox span {
    color: #eaf2f7 !important;
}

div[data-testid="stFileUploader"] {
    background: #1b2934 !important;
    border: 1px dashed #6b8797 !important;
    border-radius: 14px;
    padding: 14px;
}

div[data-testid="stFileUploader"] section {
    background: #223442 !important;
    border: 1px solid #405665 !important;
}

div[data-testid="stFileUploader"] button {
    background: #20c6b0 !important;
    color: #071b1a !important;
    border: none !important;
}

div[data-testid="stFileUploader"] small {
    color: #d0dce4 !important;
}

div[data-testid="stFileUploader"] span {
    color: #eaf2f7 !important;
}

.stButton > button,
.stDownloadButton > button,
.stFormSubmitButton > button {
    background: linear-gradient(
        100deg,
        #20c6b0,
        #16a995
    );
    color: #071b1a !important;
    border: none !important;
    border-radius: 10px;
    font-weight: 700;
    min-height: 44px;
}

.stButton > button *,
.stDownloadButton > button *,
.stFormSubmitButton > button * {
    color: #071b1a !important;
}

.stButton > button:hover,
.stDownloadButton > button:hover,
.stFormSubmitButton > button:hover {
    background: #45ddc8 !important;
    color: #071b1a !important;
    border: none !important;
}

[data-testid="stMetric"] {
    background: linear-gradient(
        145deg,
        #223746,
        #192631
    );
    border: 1px solid #405665;
    padding: 19px;
    border-radius: 15px;
    min-height: 115px;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.15);
}

[data-testid="stMetricLabel"],
[data-testid="stMetricLabel"] p {
    color: #d0dce5 !important;
    font-size: 13px;
    font-weight: 600;
}

[data-testid="stMetricValue"] {
    color: #35ddc7 !important;
    font-size: 29px;
    font-weight: 800;
}

div[data-testid="stDataFrame"] {
    border: 1px solid #405665;
    border-radius: 12px;
    overflow: hidden;
}

[data-testid="stDataFrame"] * {
    color: #edf4f8 !important;
}

[data-testid="stTable"] {
    color: #edf4f8 !important;
}

[data-testid="stTable"] th {
    background-color: #243b4a !important;
    color: #ffffff !important;
}

[data-testid="stTable"] td {
    color: #e5edf3 !important;
}

div[data-testid="stAlert"] {
    border-radius: 12px;
}

div[data-testid="stAlert"] p {
    color: #f1f6fa !important;
}

[data-testid="stExpander"] {
    background-color: #1b2935;
    border: 1px solid #405665;
    border-radius: 12px;
}

[data-testid="stExpander"] summary {
    color: #f1f6fa !important;
}

[data-testid="stExpander"] summary p {
    color: #f1f6fa !important;
}

button[data-baseweb="tab"] {
    color: #dce7ed !important;
}

button[data-baseweb="tab"][aria-selected="true"] {
    color: #35ddc7 !important;
}

hr {
    border-color: #405665 !important;
}

.hero {
    background: linear-gradient(
        115deg,
        #1c3944,
        #1b2b38 65%,
        #203b40
    );
    border: 1px solid #42616e;
    padding: 25px 28px;
    border-radius: 18px;
    margin-bottom: 24px;
}

.hero-kicker {
    color: #72eadb;
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 1.7px;
}

.hero-title {
    color: #f4fbfc;
    font-family: Manrope, sans-serif;
    font-size: 28px;
    font-weight: 800;
    margin: 7px 0;
}

.hero-copy {
    color: #d0e0e7;
    font-size: 14px;
    margin: 0;
}

.section-note {
    color: #c0ced8;
    font-size: 13px;
    margin-top: -8px;
}

.panel {
    background: #192631;
    border: 1px solid #405665;
    border-radius: 15px;
    padding: 18px;
}

.panel p {
    color: #e0eaf0 !important;
}

.pill {
    display: inline-block;
    padding: 5px 10px;
    border-radius: 20px;
    background: #153e3b;
    color: #72eadb;
    font-size: 11px;
    font-weight: 700;
}

[data-testid="stSidebar"] [data-testid="stRadio"] label {
    color: #edf4f8 !important;
}

[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
    color: #c4d1da !important;
}

[data-testid="stForm"] {
    background: #192631;
    border: 1px solid #405665;
    border-radius: 14px;
    padding: 20px;
}

[data-testid="stVegaLiteChart"] {
    background: #192631;
    border: 1px solid #405665;
    border-radius: 12px;
    padding: 10px;
}

a {
    color: #58dfd0 !important;
}

[data-testid="stTooltipIcon"] {
    color: #dce7ed !important;
}
</style>
""", unsafe_allow_html=True)


def page_hero(kicker, title, description):
    st.markdown(
        f"""
        <div class="hero">
            <div class="hero-kicker">{kicker}</div>
            <div class="hero-title">{title}</div>
            <p class="hero-copy">{description}</p>
        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# MODEL
# =========================================================

@st.cache_resource
def load_model():
    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    return YOLO(MODEL_PATH)


# =========================================================
# HISTORY MANAGEMENT
# =========================================================

def read_history_file():
    if not os.path.exists(HISTORY_FILE):
        return pd.DataFrame(columns=HISTORY_COLUMNS)

    legacy_columns = [
        "Date", "Image", "Damage Type", "Confidence"
    ]

    rows = []

    with open(
        HISTORY_FILE,
        "r",
        newline="",
        encoding="utf-8-sig"
    ) as file:

        reader = csv.reader(file)
        header = next(reader, [])

        for row in reader:
            if not row:
                continue

            record = dict.fromkeys(HISTORY_COLUMNS, "")

            if len(row) == 4:
                record.update(dict(zip(legacy_columns, row)))
            elif len(row) == 7:
                record.update(dict(zip(HISTORY_COLUMNS, row)))
            else:
                continue

            rows.append(record)

    return pd.DataFrame(rows, columns=HISTORY_COLUMNS)


def load_history():
    try:
        return read_history_file()
    except Exception as error:
        st.warning(f"History file could not be read: {error}")
        return pd.DataFrame(columns=HISTORY_COLUMNS)


def save_history(records):
    if not records:
        return

    new_df = pd.DataFrame(records)
    new_df = new_df.reindex(
        columns=HISTORY_COLUMNS,
        fill_value=""
    )

    old_df = read_history_file()
    combined = pd.concat([old_df, new_df], ignore_index=True)

    combined.to_csv(
        HISTORY_FILE,
        index=False,
        encoding="utf-8-sig"
    )


# =========================================================
# IMAGE HELPERS
# =========================================================

def save_image(image, file_path):
    image.save(file_path, format="PNG")


def image_file_available(path):
    return (
        isinstance(path, str)
        and bool(path)
        and os.path.isfile(path)
    )


# =========================================================
# PDF REPORT GENERATION
# =========================================================

def create_pdf(
    report_id,
    image_name,
    report_time,
    detections,
    original_path=None,
    annotated_path=None
):
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    dark = colors.HexColor("#171C24")
    teal = colors.HexColor("#16B8A6")
    light = colors.HexColor("#EAF8F5")
    grey = colors.HexColor("#64748B")

    def draw_footer(page_number):
        pdf.setStrokeColor(teal)
        pdf.line(40, 42, width - 40, 42)
        pdf.setFillColor(grey)
        pdf.setFont("Helvetica", 8)
        pdf.drawString(
            40, 27,
            "Civix AI – AI-Powered Road Damage Detection and Complaint Management System"
        )
        pdf.drawRightString(width - 40, 27, f"Page {page_number}")

    page_number = 1

    pdf.setFillColor(dark)
    pdf.rect(0, height - 115, width, 115, fill=1, stroke=0)

    pdf.setFillColor(teal)
    pdf.setFont("Helvetica-Bold", 19)
    pdf.drawString(40, height - 43, "CIVIX AI ROAD DAMAGE REPORT")

    pdf.setFillColor(colors.white)
    pdf.setFont("Helvetica", 9)
    pdf.drawString(
        40, height - 65,
        "Civix AI – AI-Powered Road Damage Detection and Complaint Management System"
    )

    pdf.setFillColor(colors.HexColor("#B9D8D2"))
    pdf.setFont("Helvetica", 8)
    pdf.drawString(40, height - 88, "INTELLIGENT ROAD MONITORING")

    y = height - 145

    pdf.setFillColor(light)
    pdf.roundRect(35, y - 105, width - 70, 105, 8, fill=1, stroke=0)

    pdf.setFillColor(dark)
    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(50, y - 22, "REPORT INFORMATION")

    pdf.setFillColor(colors.black)
    pdf.setFont("Helvetica", 9)
    pdf.drawString(50, y - 45, "Report ID: " + str(report_id))
    pdf.drawString(50, y - 63, "Image: " + str(image_name)[:65])
    pdf.drawString(50, y - 81, "Date & Time: " + str(report_time))

    y -= 135

    pdf.setFillColor(dark)
    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(40, y, "DETECTION SUMMARY")
    y -= 25

    pdf.setFillColor(light)
    pdf.roundRect(40, y - 43, width - 80, 48, 7, fill=1, stroke=0)

    pdf.setFillColor(teal)
    pdf.setFont("Helvetica-Bold", 18)
    pdf.drawString(55, y - 29, str(len(detections)))

    pdf.setFillColor(dark)
    pdf.setFont("Helvetica", 9)
    pdf.drawString(100, y - 27, "Total Detected Objects")

    y -= 75

    pdf.setFillColor(dark)
    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(40, y, "DETECTION DETAILS")
    y -= 23

    if detections:
        for index, item in enumerate(detections, 1):
            if y < 85:
                draw_footer(page_number)
                pdf.showPage()
                page_number += 1
                y = height - 60

            pdf.setFillColor(light)
            pdf.roundRect(40, y - 12, width - 80, 27, 5, fill=1, stroke=0)

            damage_name = str(item["class"]).replace("_", " ").title()

            pdf.setFillColor(dark)
            pdf.setFont("Helvetica-Bold", 9)
            pdf.drawString(52, y - 3, f"{index}. {damage_name}")

            pdf.setFont("Helvetica", 9)
            pdf.drawRightString(
                width - 52,
                y - 3,
                f"Confidence: {item['confidence']:.2f}%"
            )
            y -= 35
    else:
        pdf.setFillColor(grey)
        pdf.setFont("Helvetica", 10)
        pdf.drawString(50, y, "No road damage detected.")
        y -= 25

    has_original = image_file_available(original_path)
    has_annotated = image_file_available(annotated_path)

    if has_original or has_annotated:
        if y < 250:
            draw_footer(page_number)
            pdf.showPage()
            page_number += 1
            y = height - 60

        y -= 10
        pdf.setFillColor(dark)
        pdf.setFont("Helvetica-Bold", 12)
        pdf.drawString(40, y, "IMAGE ANALYSIS")
        y -= 22

        image_width = 245
        image_height = 175

        if has_original:
            pdf.setFillColor(dark)
            pdf.setFont("Helvetica-Bold", 9)
            pdf.drawString(40, y, "Original Image")
            pdf.drawImage(
                ImageReader(original_path),
                40, y - image_height - 5,
                width=image_width,
                height=image_height,
                preserveAspectRatio=True,
                anchor="c"
            )

        if has_annotated:
            pdf.setFillColor(dark)
            pdf.setFont("Helvetica-Bold", 9)
            pdf.drawString(310, y, "AI Detection Image")
            pdf.drawImage(
                ImageReader(annotated_path),
                310, y - image_height - 5,
                width=image_width,
                height=image_height,
                preserveAspectRatio=True,
                anchor="c"
            )

    draw_footer(page_number)
    pdf.save()
    buffer.seek(0)

    return buffer.getvalue()


# =========================================================
# SESSION STATE
# =========================================================

if "user" not in st.session_state:
    st.session_state.user = None

if "auth_mode" not in st.session_state:
    st.session_state.auth_mode = "Login"


# =========================================================
# SIDEBAR NAVIGATION AND AUTHENTICATION
# =========================================================

st.sidebar.markdown("""
<div style="padding:8px 0 15px 0;">
    <h2 style="color:#20c6b0 !important;margin-bottom:0;">
        🔍 CIVIX AI
    </h2>
    <span style="font-size:11px;color:#c0ced8;">
        AI-POWERED ROAD DAMAGE DETECTION
    </span>
</div>
""", unsafe_allow_html=True)

st.sidebar.divider()

user = st.session_state.user

if user:
    st.sidebar.success(
        f"Logged in: {user['name']}"
    )
    st.sidebar.caption(
        f"Account: {user['role'].title()}"
    )

    if st.sidebar.button("Logout", use_container_width=True):
        st.session_state.user = None
        st.session_state.auth_mode = "Login"
        st.rerun()

else:
    st.sidebar.info("You are browsing as a guest.")

st.sidebar.divider()

menu_items = [
    "Dashboard",
    "Detect Road Damage",
    "Reports & History",
    "Track Complaint"
]

if not user:
    menu_items.extend([
        "Login",
        "Signup"
    ])

elif user["role"] == "user":
    menu_items.extend([
        "Submit Complaint",
        "My Complaints"
    ])

elif user["role"] == "admin":
    menu_items.append("Admin Dashboard")

page = st.sidebar.radio(
    "MAIN MENU",
    menu_items,
    key="main_navigation"
)

st.sidebar.divider()
st.sidebar.markdown("**AI MODEL**")
st.sidebar.caption("YOLOv8 Deep Learning")
st.sidebar.markdown("**TECHNOLOGY**")
st.sidebar.caption("Python • Streamlit • SQLite")
st.sidebar.markdown("---")
st.sidebar.caption(PROJECT_NAME)

history = load_history()


# =========================================================
# LOGIN AND SIGNUP PAGE
# =========================================================

if page in ["Login", "Signup"]:

    page_hero(
        "USER ACCOUNT",
        page,
        "Access your account to register and track road damage complaints."
    )

    if page == "Login":

        st.markdown("### Login to Your Account")

        with st.form("login_form"):

            username = st.text_input("Username")
            password = st.text_input(
                "Password",
                type="password"
            )

            submitted = st.form_submit_button(
                "Login",
                use_container_width=True
            )

        if submitted:
            if not username or not password:
                st.error("Please enter username and password.")
            else:
                authenticated = authenticate_user(
                    username,
                    password
                )

                if authenticated:
                    st.session_state.user = authenticated
                    st.success("Login successful.")
                    st.rerun()
                else:
                    st.error("Incorrect username or password.")

        st.info(
            "To create a new account, select Signup in the sidebar."
        )

    else:

        st.markdown("### Create a New User Account")

        with st.form("signup_form"):

            name = st.text_input("Full Name")
            username = st.text_input("Choose Username")
            password = st.text_input(
                "Create Password",
                type="password"
            )
            confirm_password = st.text_input(
                "Confirm Password",
                type="password"
            )

            submitted = st.form_submit_button(
                "Create Account",
                use_container_width=True
            )

        if submitted:

            if not all([
                name.strip(),
                username.strip(),
                password,
                confirm_password
            ]):
                st.error("Please fill all fields.")

            elif len(password) < 8:
                st.error("Password must contain at least 8 characters.")

            elif password != confirm_password:
                st.error("Passwords do not match.")

            else:
                success, message = create_user(
                    name,
                    username,
                    password
                )

                if success:
                    st.success(message)
                    st.info("Now select Login from the sidebar.")
                else:
                    st.error(message)


# =========================================================
# ADMIN DASHBOARD
# =========================================================

elif page == "Admin Dashboard":

    if not user or user["role"] != "admin":
        st.error("Admin access is required.")
        st.stop()

    page_hero(
        "ADMINISTRATION",
        "Admin Dashboard",
        "Manage and review complaints submitted in this demonstration system."
    )

    complaints = get_all_complaints()

    total = len(complaints)
    pending = sum(c["status"] == "Pending" for c in complaints)
    progress = sum(c["status"] == "In Progress" for c in complaints)
    resolved = sum(c["status"] == "Resolved" for c in complaints)

    a, b, c, d = st.columns(4)
    a.metric("Total Complaints", total)
    b.metric("Pending", pending)
    c.metric("In Progress", progress)
    d.metric("Resolved", resolved)

    st.divider()
    st.markdown("### Complaint Records")

    if complaints:

        display = pd.DataFrame([
            {
                "Complaint ID": item["complaint_id"],
                "Name": item["name"],
                "Problem": item["damage_type"],
                "Location": item["location"],
                "Status": item["status"],
                "Submitted": item["created_at"]
            }
            for item in complaints
        ])

        st.dataframe(
            display,
            use_container_width=True,
            hide_index=True
        )

        st.divider()
        st.markdown("### Review and Update Complaint")

        complaint_options = {
            item["complaint_id"]: item
            for item in complaints
        }

        selected_id = st.selectbox(
            "Select Complaint ID",
            list(complaint_options.keys())
        )

        selected = complaint_options[selected_id]

        st.markdown(f"**Name:** {selected['name']}")
        st.markdown(f"**Address:** {selected['address']}")
        st.markdown(f"**Phone:** {selected['phone']}")
        st.markdown(f"**Problem:** {selected['damage_type']}")
        st.markdown(f"**Location:** {selected['location']}")
        st.markdown(f"**Description:** {selected['description']}")

        if image_file_available(selected["image_path"]):
            st.image(
                selected["image_path"],
                caption="Complaint Image",
                width=450
            )

        current_status = selected["status"]

        new_status = st.selectbox(
            "Update Complaint Status",
            COMPLAINT_STATUSES,
            index=COMPLAINT_STATUSES.index(current_status),
            key="admin_status_" + selected_id
        )

        if st.button(
            "Update Complaint Status",
            use_container_width=True
        ):
            update_complaint_status(
                selected_id,
                new_status
            )
            st.success("Complaint status updated.")
            st.rerun()

    else:
        st.info("No complaints have been submitted yet.")


# =========================================================
# PUBLIC DASHBOARD
# =========================================================

elif page == "Dashboard":

    page_hero(
        "AI-POWERED ROAD INSPECTION",
        PROJECT_NAME,
        "Using Deep Learning for automated pothole, crack and open manhole detection."
    )

    valid_damage = history[
        history["Damage Type"].isin(DAMAGE_TYPES)
    ].copy()

    if not history.empty:
        report_ids = (
            history["Report ID"].dropna().astype(str)
        )
        new_report_count = report_ids[
            report_ids.str.strip() != ""
        ].nunique()

        legacy_rows = history[
            history["Report ID"].isna()
            | (
                history["Report ID"].astype(str).str.strip() == ""
            )
        ]

        legacy_count = legacy_rows["Image"].nunique()
        total_images = new_report_count + legacy_count
    else:
        total_images = 0

    total_damages = len(valid_damage)

    potholes = int(
        (valid_damage["Damage Type"] == "pothole").sum()
    )

    cracks = int(
        (valid_damage["Damage Type"] == "cracks").sum()
    )

    manholes = int(
        (valid_damage["Damage Type"] == "open_manhole").sum()
    )

    st.markdown("### Inspection Statistics")

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric("Images Analyzed", total_images)
    c2.metric("Total Damages", total_damages)
    c3.metric("Potholes", potholes)
    c4.metric("Cracks", cracks)
    c5.metric("Open Manholes", manholes)

    left, right = st.columns([1.35, 1], gap="large")

    with left:
        st.markdown("### Damage Distribution")

        chart_data = pd.DataFrame({
            "Damage Type": [
                "Potholes",
                "Cracks",
                "Open Manholes"
            ],
            "Detections": [
                potholes,
                cracks,
                manholes
            ]
        }).set_index("Damage Type")

        st.bar_chart(
            chart_data,
            color="#20C6B0",
            height=320
        )

    with right:
        st.markdown("### Detection Summary")

        no_damage_count = int(
            (history["Damage Type"] == "No damage").sum()
        ) if not history.empty else 0

        damaged_reports = (
            history.loc[
                history["Damage Type"].isin(DAMAGE_TYPES),
                "Report ID"
            ].replace("", pd.NA).dropna().nunique()
        ) if not history.empty else 0

        st.metric("Images with Damage", damaged_reports)
        st.metric("No-Damage Records", no_damage_count)

        if total_images > 0:
            st.success("Detection history is available.")
        else:
            st.info("Upload a road image to begin your inspection.")

        st.markdown("""
        <div class="panel">
            <span class="pill">ACTIVE AI MODEL</span>
            <br><br>
            <b>YOLOv8 Road Damage Model</b>
            <p style="font-size:13px;color:#c0ced8;">
                Detects potholes, cracks and open manholes
                from uploaded road images.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.divider()
    st.markdown("### Recent Detection Activity")

    if not history.empty:
        recent = history.tail(10).iloc[::-1].copy()

        display = recent[
            [
                "Report ID",
                "Date",
                "Image",
                "Damage Type",
                "Confidence"
            ]
        ].copy()

        display["Damage Type"] = display["Damage Type"].map(
            lambda x: DAMAGE_LABELS.get(
                x,
                "No Damage" if x == "No damage"
                else str(x).replace("_", " ").title()
            )
        )

        display["Confidence"] = pd.to_numeric(
            display["Confidence"],
            errors="coerce"
        ).fillna(0).map(lambda x: f"{x:.2f}%")

        st.dataframe(
            display,
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("Your recent inspection records will appear here.")


# =========================================================
# DETECT ROAD DAMAGE
# =========================================================

elif page == "Detect Road Damage":

    page_hero(
        "AI IMAGE ANALYSIS",
        "Detect Road Damage",
        "Upload a road image and analyze it using the trained YOLOv8 deep learning model."
    )

    st.markdown("### 1. Upload Road Image")

    uploaded_file = st.file_uploader(
        "Choose a road image",
        type=["jpg", "jpeg", "png"],
        key="road_image_upload"
    )

    current_upload_id = None

    if uploaded_file is not None:

        current_upload_id = (
            uploaded_file.name,
            uploaded_file.size
        )

        try:
            image = Image.open(uploaded_file).convert("RGB")

            st.markdown("### 2. Image Preview")

            preview_col, info_col = st.columns([1.45, 1])

            with preview_col:
                st.image(
                    image,
                    caption=uploaded_file.name,
                    use_container_width=True
                )

            with info_col:
                st.markdown('<div class="panel">', unsafe_allow_html=True)
                st.markdown("**Image Information**")
                st.write(f"**File:** {uploaded_file.name}")
                st.write(
                    f"**Dimensions:** {image.width} × {image.height} px"
                )
                st.write(f"**Format:** {uploaded_file.type or 'Image'}")
                st.write("**AI Model:** YOLOv8")
                st.markdown("</div>", unsafe_allow_html=True)

            if st.button(
                "🚀 Analyze Road Image",
                type="primary",
                use_container_width=True
            ):

                try:
                    with st.spinner(
                        "AI model is analyzing the road image..."
                    ):
                        model = load_model()

                        results = model.predict(
                            source=image,
                            conf=0.25,
                            verbose=False
                        )

                        result = results[0]
                        detected_image = result.plot()[:, :, ::-1]

                        detections = []

                        for box in result.boxes:
                            class_id = int(box.cls[0])
                            confidence = float(box.conf[0])
                            class_name = result.names[class_id]

                            detections.append({
                                "class": class_name,
                                "confidence": confidence * 100
                            })

                        now = datetime.now()

                        report_time = now.strftime(
                            "%d-%m-%Y %H:%M:%S"
                        )

                        report_id = (
                            "RDI-"
                            + now.strftime("%Y%m%d%H%M%S")
                            + "-"
                            + uuid.uuid4().hex[:5].upper()
                        )

                        original_path = os.path.join(
                            IMAGE_DIR,
                            report_id + "_original.png"
                        )

                        annotated_path = os.path.join(
                            IMAGE_DIR,
                            report_id + "_annotated.png"
                        )

                        save_image(image, original_path)

                        Image.fromarray(
                            detected_image
                        ).save(
                            annotated_path,
                            format="PNG"
                        )

                        if detections:
                            records = [
                                {
                                    "Report ID": report_id,
                                    "Date": report_time,
                                    "Image": uploaded_file.name,
                                    "Damage Type": item["class"],
                                    "Confidence": round(
                                        item["confidence"], 2
                                    ),
                                    "Original Image Path": original_path,
                                    "Annotated Image Path": annotated_path
                                }
                                for item in detections
                            ]
                        else:
                            records = [{
                                "Report ID": report_id,
                                "Date": report_time,
                                "Image": uploaded_file.name,
                                "Damage Type": "No damage",
                                "Confidence": 0,
                                "Original Image Path": original_path,
                                "Annotated Image Path": annotated_path
                            }]

                        save_history(records)

                        image_buffer = io.BytesIO()

                        Image.fromarray(
                            detected_image
                        ).save(image_buffer, format="PNG")

                        st.session_state["detections"] = detections
                        st.session_state["detected_image"] = image_buffer.getvalue()
                        st.session_state["original_image_path"] = original_path
                        st.session_state["annotated_image_path"] = annotated_path
                        st.session_state["uploaded_name"] = uploaded_file.name
                        st.session_state["report_time"] = report_time
                        st.session_state["report_id"] = report_id
                        st.session_state["active_upload_id"] = current_upload_id

                    st.success(
                        "Image analysis completed and history saved successfully."
                    )

                except Exception as error:
                    st.error(f"Analysis failed: {error}")

        except Exception as error:
            st.error(f"Could not open this image: {error}")

    if (
        current_upload_id is not None
        and st.session_state.get("active_upload_id") == current_upload_id
        and "detections" in st.session_state
    ):

        detections = st.session_state["detections"]

        st.divider()
        st.markdown("### 3. AI Detection Results")

        result_left, result_right = st.columns([1.4, 1])

        with result_left:
            st.image(
                st.session_state["detected_image"],
                caption="AI Annotated Road Image",
                use_container_width=True
            )

        with result_right:
            st.metric("Total Detected Objects", len(detections))

            st.metric(
                "Potholes",
                sum(d["class"] == "pothole" for d in detections)
            )

            st.metric(
                "Cracks",
                sum(d["class"] == "cracks" for d in detections)
            )

            st.metric(
                "Open Manholes",
                sum(d["class"] == "open_manhole" for d in detections)
            )

        st.markdown("### Detection Details")

        if detections:
            detail_df = pd.DataFrame([
                {
                    "No.": i,
                    "Damage Type": DAMAGE_LABELS.get(
                        d["class"],
                        d["class"].replace("_", " ").title()
                    ),
                    "Confidence": f'{d["confidence"]:.2f}%'
                }
                for i, d in enumerate(detections, 1)
            ])

            st.dataframe(
                detail_df,
                use_container_width=True,
                hide_index=True
            )

            st.success("Road damage detected in the uploaded image.")
        else:
            st.info("No road damage detected in this image.")

        st.divider()
        show_damage_guidance(detections)

        st.divider()

        pdf_data = create_pdf(
            st.session_state["report_id"],
            st.session_state["uploaded_name"],
            st.session_state["report_time"],
            detections,
            st.session_state["original_image_path"],
            st.session_state["annotated_image_path"]
        )

        st.download_button(
            "⬇️ Download Professional PDF Report",
            data=pdf_data,
            file_name=st.session_state["report_id"] + ".pdf",
            mime="application/pdf",
            use_container_width=True
        )


# =========================================================
# REPORTS AND HISTORY
# =========================================================

elif page == "Reports & History":

    page_hero(
        "INSPECTION ARCHIVE",
        "Reports & Detection History",
        "Search, filter and download saved road inspection records and individual reports."
    )

    if not history.empty:

        total_records = len(history)

        unique_reports = history[
            "Report ID"
        ].replace("", pd.NA).nunique()

        damage_records = int(
            history["Damage Type"].isin(DAMAGE_TYPES).sum()
        )

        m1, m2, m3 = st.columns(3)

        m1.metric("Total Records", total_records)
        m2.metric("Saved Reports", unique_reports)
        m3.metric("Damage Detections", damage_records)

        st.divider()
        st.markdown("### Filter Inspection Records")

        f1, f2 = st.columns(2)

        with f1:
            damage_options = [
                "All",
                "pothole",
                "cracks",
                "open_manhole",
                "No damage"
            ]

            selected_damage = st.selectbox(
                "Damage Type",
                damage_options,
                format_func=lambda x: (
                    "All Damage Types"
                    if x == "All"
                    else (
                        "No Damage"
                        if x == "No damage"
                        else DAMAGE_LABELS.get(x, x)
                    )
                )
            )

        with f2:
            image_options = ["All"] + sorted(
                history["Image"].dropna().astype(str).unique().tolist()
            )

            selected_image = st.selectbox(
                "Image Name",
                image_options
            )

        filtered = history.copy()

        if selected_damage != "All":
            filtered = filtered[
                filtered["Damage Type"] == selected_damage
            ]

        if selected_image != "All":
            filtered = filtered[
                filtered["Image"] == selected_image
            ]

        st.divider()
        st.markdown("### Detection Records")

        if not filtered.empty:

            display = filtered.iloc[::-1][
                [
                    "Report ID",
                    "Date",
                    "Image",
                    "Damage Type",
                    "Confidence"
                ]
            ].copy()

            display["Damage Type"] = display[
                "Damage Type"
            ].map(
                lambda x: DAMAGE_LABELS.get(
                    x,
                    "No Damage" if x == "No damage"
                    else str(x).replace("_", " ").title()
                )
            )

            display["Confidence"] = pd.to_numeric(
                display["Confidence"],
                errors="coerce"
            ).fillna(0).map(lambda x: f"{x:.2f}%")

            st.dataframe(
                display,
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("No records match the selected filters.")

        st.download_button(
            "⬇️ Download Filtered History as CSV",
            data=filtered.to_csv(index=False).encode("utf-8-sig"),
            file_name="Road_Damage_History.csv",
            mime="text/csv"
        )

        st.caption(
            f"Showing {len(filtered)} record(s) after applying filters."
        )

        st.divider()
        st.markdown("### Individual PDF Report")

        report_options = {}

        for report_id in filtered["Report ID"].dropna().unique():
            if str(report_id).strip():
                report_rows = filtered[
                    filtered["Report ID"] == report_id
                ]

                first = report_rows.iloc[0]

                label = (
                    f'{first["Image"]} | '
                    f'{first["Date"]} | '
                    f'{report_id}'
                )

                report_options[label] = report_id

        if report_options:

            selected_label = st.selectbox(
                "Select an Inspection Report",
                list(report_options.keys())
            )

            selected_report_id = report_options[selected_label]

            report_rows = history[
                history["Report ID"] == selected_report_id
            ]

            first = report_rows.iloc[0]
            report_detections = []

            for _, row in report_rows.iterrows():
                if row["Damage Type"] in DAMAGE_TYPES:
                    try:
                        confidence = float(row["Confidence"])
                    except (TypeError, ValueError):
                        confidence = 0.0

                    report_detections.append({
                        "class": row["Damage Type"],
                        "confidence": confidence
                    })

            pdf_data = create_pdf(
                selected_report_id,
                str(first["Image"]),
                str(first["Date"]),
                report_detections,
                first["Original Image Path"],
                first["Annotated Image Path"]
            )

            st.download_button(
                "⬇️ Download Selected PDF Report",
                data=pdf_data,
                file_name=str(selected_report_id) + ".pdf",
                mime="application/pdf"
            )
        else:
            st.info(
                "Individual PDF downloads are available for newly generated reports."
            )

    else:
        st.info(
            "No reports are available yet. Analyze a road image to create your first report."
        )


# =========================================================
# SUBMIT COMPLAINT
# =========================================================

elif page == "Submit Complaint":

    if not user or user["role"] != "user":
        st.error("Please login with a user account to submit a complaint.")
        st.stop()

    page_hero(
        "CITIZEN REPORTING",
        "Submit Road Damage Complaint",
        "Register a road-related complaint for review in this demonstration system."
    )

    st.info(
        "This is an independent project simulation. "
        "Your complaint will be stored in this application's database, "
        "not sent to an actual Nagar Nigam."
    )

    with st.form("complaint_form", clear_on_submit=True):

        st.markdown("### Personal Information")

        name = st.text_input(
            "Full Name",
            value=user["name"]
        )

        address = st.text_area("Address")

        phone = st.text_input(
            "Phone Number",
            max_chars=15
        )

        st.markdown("### Road Problem Details")

        damage_type = st.selectbox(
            "Problem Type",
            [
                "Pothole",
                "Road Cracks",
                "Open Manhole",
                "Other"
            ]
        )

        location = st.text_input(
            "Location of Road Problem",
            placeholder="Enter road, area or landmark"
        )

        description = st.text_area(
            "Describe the Problem",
            placeholder="Explain the road problem and its location..."
        )

        complaint_image = st.file_uploader(
            "Upload Road Image (Optional)",
            type=["jpg", "jpeg", "png"],
            key="complaint_image"
        )

        submitted = st.form_submit_button(
            "Submit Complaint",
            use_container_width=True
        )

    if submitted:

        if not all([
            name.strip(),
            address.strip(),
            phone.strip(),
            location.strip(),
            description.strip()
        ]):
            st.error("Please fill all required fields.")

        elif not phone.strip().replace("+", "").isdigit() or len(
            phone.strip().replace("+", "")
        ) < 10:
            st.error("Please enter a valid phone number.")

        else:

            image_path = None

            try:

                if complaint_image is not None:

                    image = Image.open(
                        complaint_image
                    ).convert("RGB")

                    image_path = os.path.join(
                        COMPLAINT_IMAGE_DIR,
                        uuid.uuid4().hex + ".png"
                    )

                    image.save(image_path, format="PNG")

                complaint_id = create_complaint(
                    user["id"],
                    name,
                    address,
                    phone,
                    damage_type,
                    location,
                    description,
                    image_path
                )

                st.success("Complaint submitted successfully!")

                st.markdown(
                    f"### Your Complaint ID: `{complaint_id}`"
                )

                st.warning(
                    "Please save this Complaint ID to check your status later."
                )

            except Exception as error:
                st.error(f"Could not submit complaint: {error}")


# =========================================================
# MY COMPLAINTS
# =========================================================

elif page == "My Complaints":

    if not user or user["role"] != "user":
        st.error("Please login first.")
        st.stop()

    page_hero(
        "USER ACCOUNT",
        "My Complaints",
        "View the complaints you have submitted and their latest status."
    )

    complaints = get_user_complaints(user["id"])

    if complaints:

        display = pd.DataFrame([
            {
                "Complaint ID": c["complaint_id"],
                "Problem": c["damage_type"],
                "Location": c["location"],
                "Status": c["status"],
                "Submitted": c["created_at"],
                "Last Updated": c["updated_at"]
            }
            for c in complaints
        ])

        st.dataframe(
            display,
            use_container_width=True,
            hide_index=True
        )

    else:
        st.info("You have not submitted any complaints yet.")


# =========================================================
# PUBLIC COMPLAINT TRACKING
# =========================================================

elif page == "Track Complaint":

    page_hero(
        "COMPLAINT TRACKING",
        "Track Your Complaint",
        "Enter your Complaint ID to check the latest status."
    )

    complaint_id = st.text_input(
        "Enter Complaint ID",
        placeholder="Example: CMP-20261101-ABC1234"
    )

    if st.button(
        "Check Complaint Status",
        use_container_width=True
    ):

        if not complaint_id.strip():
            st.warning("Please enter a Complaint ID.")

        else:

            complaint = get_complaint_by_id(complaint_id)

            if complaint:

                st.success("Complaint record found.")

                st.markdown(
                    f"### Complaint ID: `{complaint['complaint_id']}`"
                )

                st.write(
                    f"**Problem Type:** {complaint['damage_type']}"
                )

                st.write(
                    f"**Submitted:** {complaint['created_at']}"
                )

                st.write(
                    f"**Last Updated:** {complaint['updated_at']}"
                )

                status = complaint["status"]

                if status == "Pending":
                    st.warning("Status: Pending")
                elif status == "In Progress":
                    st.info("Status: In Progress")
                else:
                    st.success("Status: Resolved")

            else:
                st.error("No complaint found with this ID.")


# =========================================================
# ADMIN ACCOUNT INITIAL SETUP
# =========================================================

if (
    not admin_exists()
    and not st.session_state.get("admin_setup_done", False)
):
    st.sidebar.divider()

    st.sidebar.warning(
        "Admin account has not been created yet."
    )

    with st.sidebar.expander("Create First Admin Account"):

        with st.form("first_admin_setup"):

            admin_name = st.text_input("Admin Name")
            admin_username = st.text_input("Admin Username")
            admin_password = st.text_input(
                "Admin Password",
                type="password"
            )
            admin_confirm = st.text_input(
                "Confirm Admin Password",
                type="password"
            )

            create_admin = st.form_submit_button(
                "Create Admin Account"
            )

        if create_admin:

            if not all([
                admin_name.strip(),
                admin_username.strip(),
                admin_password,
                admin_confirm
            ]):
                st.error("Please fill all admin fields.")

            elif len(admin_password) < 12:
                st.error(
                    "Admin password must contain at least 12 characters."
                )

            elif admin_password != admin_confirm:
                st.error("Admin passwords do not match.")

            else:

                success, message = create_user(
                    admin_name,
                    admin_username,
                    admin_password,
                    role="admin"
                )

                if success:
                    st.session_state.admin_setup_done = True
                    st.success(
                        "Admin account created. Login using Admin username."
                    )
                    st.rerun()
                else:
                    st.error(message)


# =========================================================
# FOOTER
# =========================================================

st.sidebar.caption(
    "Demonstration project only. Complaints are not sent to a government authority."
)