"""Aplikasi Streamlit untuk deteksi emosi wajah dengan ConvNeXt Tiny."""

from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import timm
import torch
from PIL import Image
from torchvision import transforms


BASE_DIR = Path(__file__).resolve().parent
WEIGHTS_PATH = BASE_DIR / "convnext_ferplus_statedict.pth"
CLASS_NAMES = [
	"neutral", "happiness", "surprise", "sadness",
	"anger", "disgust", "fear", "contempt",
]
IMG_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

EMOTION_UI = {
	"neutral": ("Netral", "😐", "Wajah terlihat tenang dan santai.", "#78909c"),
	"happiness": ("Senang", "😄", "Wah, kamu terlihat senang!", "#ffc107"),
	"surprise": ("Terkejut", "😲", "Ekspresi terkejut terdeteksi.", "#ff7043"),
	"sadness": ("Sedih", "😢", "Kamu terlihat sedikit sedih.", "#42a5f5"),
	"anger": ("Marah", "😠", "Ekspresi marah terdeteksi.", "#e53935"),
	"disgust": ("Jijik", "🤢", "Ekspresi jijik terdeteksi.", "#7cb342"),
	"fear": ("Takut", "😨", "Ekspresi takut terdeteksi.", "#ab47bc"),
	"contempt": ("Meremehkan", "😏", "Ekspresi meremehkan terdeteksi.", "#8d6e63"),
}

st.set_page_config(page_title="Deteksi Emosi Wajah", page_icon="🎭", layout="wide")
st.markdown(
	"""
	<style>
	.stApp { background: #e9eff7; color: #172033; }
	[data-testid="stAppViewContainer"] { background: #e9eff7; }
	.block-container { max-width: 1180px; padding-top: 2rem; padding-bottom: 3rem; }
	.hero { padding: 1.5rem 1.75rem; border-radius: 14px; background: #2447b8; color: white; margin-bottom: 1.5rem; box-shadow: 0 8px 24px rgba(36, 71, 184, .16); }
	.hero h1 { margin: 0; font-size: 2rem; }
	.hero p { margin: .35rem 0 0; opacity: .9; }
	[data-testid="stHeader"] { background: transparent; }
	[data-testid="stWidgetLabel"] p, [data-testid="stMarkdownContainer"] p, .stSubheader { color: #172033 !important; }
	[data-testid="stFileUploader"] section { background: #ffffff; border: 1px solid #c7d2e2; border-radius: 12px; padding: .8rem; box-shadow: 0 5px 16px rgba(35, 55, 85, .07); }
	[data-testid="stFileUploader"] section:hover { border-color: #4d8edb; }
	[data-testid="stFileUploaderDropzoneInstructions"] div, [data-testid="stFileUploaderDropzoneInstructions"] small { color: #52606d !important; }
	[data-testid="stFileUploader"] button { color: #ffffff !important; background: #3155d6 !important; border: 0 !important; }
	[data-testid="stFileUploader"] svg { color: #3155d6 !important; fill: #3155d6 !important; }
	[data-testid="stCameraInput"] { color: #172033; }
	.result { max-width: 560px; padding: 1rem 1.2rem; border-left: 8px solid #3155d6; border-radius: 12px; background: #ffffff; margin: .75rem auto 1.25rem; box-shadow: 0 8px 22px rgba(35, 55, 85, .10); text-align: center; }
	.result-title { color: #172033 !important; font-size: 1.35rem; font-weight: 700; }
	.result-message { color: #52606d !important; margin-top: .3rem; }
	.probability-panel { max-width: 680px; margin: 0 auto; padding: 1.15rem 1.25rem; border-radius: 12px; background: #ffffff; box-shadow: 0 8px 22px rgba(35, 55, 85, .10); }
	.probability-title { color: #172033; text-align: center; font-size: 1.1rem; font-weight: 700; margin-bottom: 1rem; }
	.probability-row { display: grid; grid-template-columns: 132px 1fr 52px; align-items: center; gap: 12px; margin: .65rem 0; color: #26344a; font-size: .92rem; }
	.probability-label { text-align: right; white-space: nowrap; }
	.probability-track { height: 11px; overflow: hidden; border-radius: 99px; background: #e2e8f0; }
	.probability-bar { height: 100%; border-radius: inherit; background: #4d8edb; }
	.probability-value { color: #52606d; font-variant-numeric: tabular-nums; }
	@media (max-width: 640px) { .probability-row { grid-template-columns: 112px 1fr 48px; gap: 8px; font-size: .82rem; } .hero h1 { font-size: 1.6rem; } }
	</style>
	<div class="hero"><h1>🎭 Tebak Emosimu!</h1><p>Sistem deteksi emosi wajah berbasis ConvNeXt Tiny dan FER+</p></div>
	""",
	unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Memuat model ConvNeXt Tiny...")
def load_model():
	if not WEIGHTS_PATH.exists():
		raise FileNotFoundError(f"File model tidak ditemukan: {WEIGHTS_PATH.name}")
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model = timm.create_model("convnext_tiny", pretrained=False, num_classes=len(CLASS_NAMES))
	state = torch.load(WEIGHTS_PATH, map_location="cpu")
	if isinstance(state, dict) and "state_dict" in state:
		state = state["state_dict"]
	if isinstance(state, dict) and "model_state_dict" in state:
		state = state["model_state_dict"]
	state = {
		key.replace("_orig_mod.", "", 1).replace("module.", "", 1): value
		for key, value in state.items()
	}
	model.load_state_dict(state)
	return model.to(device).eval(), device


@st.cache_resource
def get_preprocess():
	return transforms.Compose([
		transforms.Resize((IMG_SIZE, IMG_SIZE)),
		transforms.ToTensor(),
		transforms.Normalize(MEAN, STD),
	])


@st.cache_resource
def get_face_cascade():
	cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
	if cascade.empty():
		raise RuntimeError("Haar cascade untuk deteksi wajah gagal dimuat.")
	return cascade


def detect_emotion(image):
	model, device = load_model()
	image_array = np.array(image.convert("RGB"))
	gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
	faces = get_face_cascade().detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
	if len(faces) == 0:
		return image_array, None

	x, y, width, height = max(faces, key=lambda face: face[2] * face[3])
	margin = int(0.1 * max(width, height))
	x0, y0 = max(x - margin, 0), max(y - margin, 0)
	x1, y1 = min(x + width + margin, image_array.shape[1]), min(y + height + margin, image_array.shape[0])
	face = image_array[y0:y1, x0:x1]
	face_gray = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY)
	tensor = get_preprocess()(Image.fromarray(face_gray).convert("RGB")).unsqueeze(0).to(device)
	with torch.inference_mode():
		probabilities = torch.softmax(model(tensor), dim=1)[0].cpu().numpy()

	annotated = image_array.copy()
	cv2.rectangle(annotated, (x, y), (x + width, y + height), (49, 85, 214), 4)
	return annotated, probabilities


try:
	uploaded_file = st.file_uploader("Unggah foto wajah", type=["jpg", "jpeg", "png"])
	camera_file = st.camera_input("Atau ambil foto dari kamera")
	image_file = camera_file or uploaded_file

	if image_file is None:
		st.info("Unggah foto atau ambil gambar lewat kamera untuk memulai deteksi.")
	else:
		image = Image.open(image_file).convert("RGB")
		with st.spinner("Menganalisis ekspresi wajah..."):
			annotated, probabilities = detect_emotion(image)
		st.subheader("Hasil deteksi")
		preview_left, preview_center, preview_right = st.columns([1, 2, 1])
		with preview_center:
			st.image(annotated, use_container_width=True)
		if probabilities is None:
			st.warning("Wajah tidak ditemukan. Coba foto dengan pencahayaan yang lebih baik.")
		else:
			top_index = int(np.argmax(probabilities))
			emotion = CLASS_NAMES[top_index]
			label, emoji, message, color = EMOTION_UI[emotion]
			confidence = float(probabilities[top_index]) * 100
			st.markdown(
				f'<div class="result" style="border-left-color: {color}"><div class="result-title">{emoji} {label} ({confidence:.1f}%)</div><div class="result-message">{message}</div></div>',
				unsafe_allow_html=True,
			)
			scores = {
				f"{EMOTION_UI[name][1]} {EMOTION_UI[name][0]}": float(score)
				for name, score in zip(CLASS_NAMES, probabilities)
			}
			probability_rows = []
			for name, score in scores.items():
				percentage = score * 100
				probability_rows.append(
					f'<div class="probability-row">'
					f'<div class="probability-label">{name}</div>'
					f'<div class="probability-track"><div class="probability-bar" style="width: {percentage:.2f}%"></div></div>'
					f'<div class="probability-value">{percentage:.1f}%</div>'
					f'</div>'
				)
			st.markdown(
				'<div class="probability-panel">'
				'<div class="probability-title">Probabilitas setiap emosi</div>'
				+ "".join(probability_rows)
				+ "</div>",
				unsafe_allow_html=True,
			)
except (FileNotFoundError, RuntimeError) as error:
	st.error(str(error))
