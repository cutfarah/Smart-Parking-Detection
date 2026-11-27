
from pathlib import Path
import io

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

# 1. CONFIGURATION
ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "best.pt"

st.set_page_config(
    page_title="Smart Parking Detection",
    page_icon="🅿️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# 2. CUSTOM CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background-color: #F4F7FC;
    }

    [data-testid="stSidebar"] {
        background-color: #102B50;
    }

    [data-testid="stSidebar"] * {
        color: white !important;
    }

    .main-header {
        background: linear-gradient(120deg, #102B50, #1769AA);
        padding: 30px 35px;
        border-radius: 18px;
        color: white;
        margin-bottom: 25px;
    }

    .main-header h1 {
        color: white;
        font-size: 32px;
        font-weight: 800;
        margin-bottom: 8px;
    }

    .main-header p {
        color: #DCEBFA;
        font-size: 14px;
        margin-bottom: 0;
    }

    .section-title {
        color: #17365D;
        font-size: 20px;
        font-weight: 750;
        margin-top: 15px;
        margin-bottom: 12px;
    }

    .stat-card {
        background: white;
        padding: 20px;
        border-radius: 14px;
        border: 1px solid #E3EAF3;
        box-shadow: 0 3px 12px rgba(20,50,90,0.04);
        min-height: 112px;
    }

    .stat-label {
        color: #718096;
        font-size: 13px;
        font-weight: 600;
        margin-bottom: 10px;
    }

    .stat-number {
        font-size: 31px;
        font-weight: 800;
        color: #17365D;
    }

    .stat-total { color: #1769AA; }
    .stat-occupied { color: #E04B55; }
    .stat-empty { color: #15966B; }

    .image-card {
        background: white;
        padding: 15px;
        border: 1px solid #E3EAF3;
        border-radius: 14px;
    }

    .image-heading {
        color: #17365D;
        font-size: 15px;
        font-weight: 700;
        margin-bottom: 12px;
    }

    .footer {
        text-align: center;
        color: #8795A8;
        font-size: 12px;
        padding: 25px 0 10px 0;
    }

    div.stButton > button {
        background: #1769AA;
        color: white;
        border-radius: 9px;
        border: none;
        font-weight: 600;
    }

    div.stButton > button:hover {
        background: #102B50;
        color: white;
    }
</style>
""", unsafe_allow_html=True)


# 3. LOAD MODEL
@st.cache_resource
def load_model(model_path):
    return YOLO(model_path)


# 4. DETECTION FUNCTION
def normalize_label(label):
    return (
        str(label)
        .strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
    )


def detect_parking(image, model, confidence):

    image_rgb = np.array(image.convert("RGB"))

    result = model.predict(
        source=image_rgb,
        conf=confidence,
        verbose=False
    )[0]

    annotated = Image.fromarray(image_rgb.copy())
    draw = ImageDraw.Draw(annotated)

    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except OSError:
        font = ImageFont.load_default()

    empty_count = 0
    occupied_count = 0
    details = []

    for box in result.boxes:

        class_id = int(box.cls[0].item())
        label_raw = result.names[class_id]
        label = normalize_label(label_raw)

        score = float(box.conf[0].item())

        x1, y1, x2, y2 = map(
            int, box.xyxy[0].tolist()
        )

        # Status slot parkir

        if label in {"space_empty", "empty", "vacant"}:

            status = "KOSONG"
            color = (21, 150, 107)
            empty_count += 1

        elif label in {"space_occupied", "occupied"}:

            status = "TERISI"
            color = (224, 75, 85)
            occupied_count += 1

        else:
            continue

        # Bounding box

        draw.rectangle(
            [(x1, y1), (x2, y2)],
            outline=color,
            width=4
        )

        # Label background

        label_text = f"{status} {score:.2f}"

        bbox = draw.textbbox(
            (0, 0),
            label_text,
            font=font
        )

        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        label_y = max(0, y1 - text_height - 12)

        draw.rounded_rectangle(
            [
                (x1, label_y),
                (x1 + text_width + 14, label_y + text_height + 10)
            ],
            radius=5,
            fill=color
        )

        draw.text(
            (x1 + 7, label_y + 4),
            label_text,
            fill=(255, 255, 255),
            font=font
        )

        details.append({
            "No.": len(details) + 1,
            "Status Slot": status,
            "Kelas Model": str(label_raw),
            "Confidence": round(score, 3)
        })

    total = empty_count + occupied_count

    return (
        annotated,
        total,
        occupied_count,
        empty_count,
        details
    )


# 5. SIDEBAR
with st.sidebar:

    st.markdown(
        "<h2 style='color:white;'>🅿️ Smart Parking</h2>",
        unsafe_allow_html=True
    )

    st.caption("COMPUTER VISION SYSTEM")

    st.divider()

    st.markdown("### ⚙️ Detection Settings")

    confidence = st.slider(
        "Confidence Threshold",
        min_value=0.05,
        max_value=0.95,
        value=0.25,
        step=0.05
    )

    st.caption(
        "Mengatur tingkat keyakinan model "
        "dalam mendeteksi slot parkir."
    )

    st.divider()

    st.markdown("### 🤖 Model Information")

    st.write("Model: YOLO11")
    st.write("Dataset: PKLot")
    st.write("Classes:")
    st.write("- Space Empty")
    st.write("- Space Occupied")

    st.divider()

    st.caption("Smart Parking Detection | UAS Computer Vision")


# 6. MAIN HEADER
st.markdown("""
<div class="main-header">
    <h1>Smart Parking Detection</h1>
    <p>
        Intelligent parking occupancy detection system
        powered by YOLO11 and Computer Vision.
    </p>
</div>
""", unsafe_allow_html=True)


# 7. MODEL CHECK
if not MODEL_PATH.exists():

    st.error("Model best.pt belum ditemukan!")

    st.info(
        "Pastikan model hasil training sudah tersedia "
        "di dalam folder models."
    )

    st.code(
        "smart-parking-detection/\n"
        "├── app.py\n"
        "├── requirements.txt\n"
        "└── models/\n"
        "    └── best.pt"
    )

    st.stop()


try:
    model = load_model(str(MODEL_PATH))

except Exception as exc:

    st.error(f"Model gagal dimuat: {exc}")
    st.stop()


# 8. UPLOAD SECTION
st.markdown(
    '<div class="section-title">📤 Upload Parking Image</div>',
    unsafe_allow_html=True
)

st.write(
    "Upload gambar area parkir untuk mendeteksi "
    "slot yang kosong dan terisi secara otomatis."
)

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"],
    help="Format gambar: JPG, JPEG, PNG"
)


# 9. PROCESS IMAGE
if uploaded_file is None:

    st.info(
        "👆 Silakan upload gambar area parkir "
        "untuk memulai proses deteksi."
    )

else:

    try:
        original_image = Image.open(uploaded_file).convert("RGB")

    except Exception as exc:

        st.error(f"Gambar tidak dapat dibuka: {exc}")
        st.stop()

    with st.spinner("Sedang menganalisis area parkir..."):

        try:

            (
                output_image,
                total,
                occupied,
                empty,
                details
            ) = detect_parking(
                original_image,
                model,
                confidence
            )

        except Exception as exc:

            st.error(f"Proses deteksi gagal: {exc}")
            st.stop()


    # 10. IMAGE COMPARISON
    st.markdown(
        '<div class="section-title">🖼️ Detection Results</div>',
        unsafe_allow_html=True
    )

    left, right = st.columns(2, gap="medium")

    with left:

        st.markdown(
            '<div class="image-heading">Original Image</div>',
            unsafe_allow_html=True
        )

        st.image(
            original_image,
            use_container_width=True
        )

    with right:

        st.markdown(
            '<div class="image-heading">Detection Image</div>',
            unsafe_allow_html=True
        )

        st.image(
            output_image,
            use_container_width=True
        )


    # 11. PARKING STATISTICS
    st.markdown(
        '<div class="section-title">📊 Parking Availability</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">TOTAL PARKING SLOTS</div>
            <div class="stat-number stat-total">{total}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:

        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">OCCUPIED SLOTS</div>
            <div class="stat-number stat-occupied">{occupied}</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:

        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-label">AVAILABLE SLOTS</div>
            <div class="stat-number stat-empty">{empty}</div>
        </div>
        """, unsafe_allow_html=True)


    # 12. OCCUPANCY RATE
    st.markdown(
        '<div class="section-title">📈 Parking Occupancy Rate</div>',
        unsafe_allow_html=True
    )

    if total > 0:

        occupancy = occupied / total

        st.progress(occupancy)

        st.write(
            f"Parking occupancy: **{occupancy * 100:.1f}%**"
        )

        st.write(
            f"Available capacity: **{100 - occupancy * 100:.1f}%**"
        )

    else:

        st.warning(
            "Tidak ditemukan slot parkir. "
            "Coba gambar lain atau sesuaikan confidence threshold."
        )


    # 13. DETECTION DETAILS
    st.markdown(
        '<div class="section-title">📋 Detection Details</div>',
        unsafe_allow_html=True
    )

    if details:

        st.dataframe(
            details,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.info("Belum ada slot parkir yang terdeteksi.")

    # 14. DOWNLOAD RESULT
    st.markdown(
        '<div class="section-title">⬇️ Download Result</div>',
        unsafe_allow_html=True
    )

    image_buffer = io.BytesIO()

    output_image.save(
        image_buffer,
        format="PNG"
    )

    st.download_button(
        label="⬇️ Download Detection Image",
        data=image_buffer.getvalue(),
        file_name="smart_parking_result.png",
        mime="image/png",
        use_container_width=True
    )

# 15. FOOTER
st.markdown("""
<div class="footer">
    SMART PARKING DETECTION SYSTEM<br>
    Computer Vision Project | YOLO11 + OpenCV + Streamlit
</div>
""", unsafe_allow_html=True)
