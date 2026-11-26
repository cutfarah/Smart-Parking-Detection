from pathlib import Path
import io

import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO

# -------------------------------------------------
# KONFIGURASI
# -------------------------------------------------
ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "best.pt"

st.set_page_config(
    page_title="Smart Parking Detection",
    page_icon="🅿️",
    layout="wide",
)

st.title("🅿️ Smart Parking Detection")
st.write(
    "Sistem deteksi slot parkir kosong dan terisi menggunakan model YOLO11 "
    "yang dilatih dengan dataset PKLot."
)

# -------------------------------------------------
# LOAD MODEL
# -------------------------------------------------
@st.cache_resource
def load_model(model_path: str):
    return YOLO(model_path)


if not MODEL_PATH.exists():
    st.error(
        "File model belum ditemukan. Letakkan model hasil training bernama "
        "`best.pt` di dalam folder `models/`."
    )
    st.code("smart-parking-detection/\n├── app.py\n├── requirements.txt\n└── models/\n    └── best.pt")
    st.stop()

try:
    model = load_model(str(MODEL_PATH))
except Exception as exc:
    st.error(f"Model gagal dimuat: {exc}")
    st.stop()

# -------------------------------------------------
# FUNGSI DETEKSI
# -------------------------------------------------
def normalize_label(label):
    return str(label).strip().lower().replace("-", "_").replace(" ", "_")


def detect_parking(image: Image.Image, confidence: float):
    image_rgb = np.array(image.convert("RGB"))

    result = model.predict(
        source=image_rgb,
        conf=confidence,
        verbose=False,
    )[0]

    annotated = image_rgb.copy()
    empty_count = 0
    occupied_count = 0
    details = []

    # Nama kelas sesuai notebook:
    # class 0 = space-empty
    # class 1 = space-occupied
    for box in result.boxes:
        class_id = int(box.cls[0].item())
        label_raw = result.names[class_id]
        label = normalize_label(label_raw)
        score = float(box.conf[0].item())

        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

        if label in {"space_empty", "empty"}:
            status = "KOSONG"
            color = (0, 180, 0)  # RGB hijau
            empty_count += 1
        elif label in {"space_occupied", "occupied"}:
            status = "TERISI"
            color = (220, 40, 40)  # RGB merah
            occupied_count += 1
        else:
            # Abaikan kelas yang tidak termasuk dua kelas parkir.
            continue

        # Gambar bounding box dan label pada gambar RGB.
        # PIL/Streamlit membaca warna dalam urutan RGB.
        import cv2
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            annotated,
            f"{status} {score:.2f}",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )

        details.append(
            {
                "Status slot": status,
                "Kelas model": str(label_raw),
                "Confidence": round(score, 3),
            }
        )

    total = empty_count + occupied_count
    return annotated, total, occupied_count, empty_count, details


# -------------------------------------------------
# ANTARMUKA STREAMLIT
# -------------------------------------------------
with st.sidebar:
    st.header("Pengaturan deteksi")
    confidence = st.slider(
        "Confidence threshold",
        min_value=0.05,
        max_value=0.95,
        value=0.25,
        step=0.05,
        help="Semakin tinggi nilainya, semakin ketat model menerima prediksi.",
    )
    st.caption("Model: YOLO11 hasil training PKLot")
    st.caption("Kelas: space-empty dan space-occupied")

uploaded_file = st.file_uploader(
    "Upload gambar area parkir",
    type=["jpg", "jpeg", "png"],
    help="Unggah satu gambar area parkir untuk mendeteksi slot kosong dan terisi.",
)

if uploaded_file is None:
    st.info("Silakan upload gambar parkir untuk memulai deteksi.")
    st.stop()

try:
    original_image = Image.open(uploaded_file).convert("RGB")
except Exception as exc:
    st.error(f"Gambar tidak dapat dibuka: {exc}")
    st.stop()

with st.spinner("Model sedang mendeteksi slot parkir..."):
    try:
        output_image, total, occupied, empty, details = detect_parking(
            original_image,
            confidence,
        )
    except Exception as exc:
        st.error(f"Terjadi kesalahan saat deteksi: {exc}")
        st.stop()

left, right = st.columns(2)
with left:
    st.subheader("Gambar asli")
    st.image(original_image, use_container_width=True)

with right:
    st.subheader("Hasil deteksi")
    st.image(output_image, use_container_width=True)

st.subheader("Ringkasan ketersediaan parkir")
metric1, metric2, metric3 = st.columns(3)
metric1.metric("Total slot terdeteksi", total)
metric2.metric("Slot terisi", occupied)
metric3.metric("Slot kosong", empty)

if total > 0:
    occupancy = occupied / total
    st.progress(occupancy)
    st.caption(f"Tingkat keterisian: {occupancy * 100:.1f}%")
else:
    st.warning(
        "Model belum mendeteksi slot parkir pada gambar ini. "
        "Coba gambar dengan area parkir yang lebih jelas atau sesuaikan confidence threshold."
    )

if details:
    st.subheader("Detail hasil deteksi")
    st.dataframe(details, use_container_width=True, hide_index=True)

image_buffer = io.BytesIO()
Image.fromarray(output_image).save(image_buffer, format="PNG")
st.download_button(
    label="Download gambar hasil deteksi",
    data=image_buffer.getvalue(),
    file_name="smart_parking_result.png",
    mime="image/png",
)
