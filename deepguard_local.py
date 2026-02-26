import streamlit as st
import torch
import numpy as np
import cv2
from torchvision import models
from facenet_pytorch import MTCNN
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# ---------------- CONFIG ----------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FRAME_SAMPLE_COUNT = 12

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    model = models.mobilenet_v2(pretrained=True)
    model.eval()
    model.to(DEVICE)
    return model

model = load_model()

# ---------------- FACE DETECTOR ----------------
mtcnn = MTCNN(keep_all=False, device=DEVICE)

# ---------------- FRAME SAMPLING ----------------
def sample_frames(video_path, n=FRAME_SAMPLE_COUNT):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = max(total // n, 1)
    frames = []

    for i in range(0, total, interval):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        if len(frames) >= n:
            break

    cap.release()
    return frames

# ---------------- GRAD CAM ----------------
def generate_gradcam(face_tensor):
    target_layers = [model.features[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)

    grayscale_cam = cam(input_tensor=face_tensor)[0]
    face_np = face_tensor.squeeze().permute(1,2,0).cpu().numpy()
    face_np = np.clip(face_np, 0, 1)

    heatmap = show_cam_on_image(face_np, grayscale_cam, use_rgb=True)
    return heatmap

# ---------------- ANALYSIS ----------------
def analyze_video(video_path):

    frames = sample_frames(video_path)

    sharpness_scores = []
    frame_diffs = []
    frequency_scores = []
    cnn_scores = []   # NEW
    heatmap_output = None

    prev_face_gray = None

    for frame in frames:

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face = mtcnn(rgb)

        if face is None:
            continue

        face_tensor = face.unsqueeze(0).to(DEVICE)

        # ---------------- CNN FAKE PROBABILITY ----------------
        with torch.no_grad():
            output = model(face_tensor)
            prob = torch.sigmoid(output).item()
            cnn_scores.append(prob)

        # Convert to numpy for statistical analysis
        face_np = face.squeeze().permute(1,2,0).cpu().numpy()
        face_np = (face_np * 255).astype(np.uint8)
        gray = cv2.cvtColor(face_np, cv2.COLOR_RGB2GRAY)

        # 1️⃣ Sharpness (Spatial)
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        sharpness_scores.append(sharpness)

        # 2️⃣ Temporal Difference
        if prev_face_gray is not None:
            diff = np.mean(np.abs(gray - prev_face_gray))
            frame_diffs.append(diff)

        prev_face_gray = gray

        # 3️⃣ Frequency Domain Analysis (FFT)
        fft = np.fft.fft2(gray)
        fft_shift = np.fft.fftshift(fft)
        magnitude = np.log(np.abs(fft_shift) + 1)
        frequency_energy = np.mean(magnitude)
        frequency_scores.append(frequency_energy)

        # Generate one heatmap for demo
        if heatmap_output is None:
            heatmap_output = generate_gradcam(face_tensor)

    if len(sharpness_scores) == 0:
        return {"error": "No face detected"}

    # ---------------- AGGREGATION ----------------
    sharpness_var = np.var(sharpness_scores)
    motion_var = np.var(frame_diffs) if len(frame_diffs) > 0 else 0
    frequency_var = np.var(frequency_scores) if len(frequency_scores) > 0 else 0
    cnn_mean = np.mean(cnn_scores) if len(cnn_scores) > 0 else 0

    # ---------------- NORMALIZATION ----------------
    sharpness_norm = sharpness_var / (sharpness_var + 100000)
    motion_norm = motion_var / (motion_var + 500)
    frequency_norm = frequency_var / (frequency_var + 1000)

    # ---------------- FINAL HYBRID FUSION ----------------
    final_score = (
        0.4 * cnn_mean +          # CNN Deepfake Probability
        0.25 * sharpness_norm +   # Spatial
        0.2 * motion_norm +       # Temporal
        0.15 * frequency_norm     # Frequency
    )

    # ---------------- LABEL ----------------
    if final_score < 0.4:
        label = "🟢 Likely Authentic"
    elif final_score < 0.7:
        label = "🟡 Inconclusive"
    else:
        label = "🔴 High Deepfake Risk"

    return {
        "score": round(final_score, 3),
        "label": label,
        "cnn_score": round(cnn_mean, 3),
        "sharpness_var": round(sharpness_var, 3),
        "motion_var": round(motion_var, 3),
        "frequency_var": round(frequency_var, 3),
        "heatmap": heatmap_output
    }
# ---------------- UI ----------------
st.title("🛡 DeepGuard – Hybrid Deepfake Detection Prototype")

uploaded_video = st.file_uploader("Upload Video", type=["mp4", "mov", "avi"])

if uploaded_video is not None:
    with open("temp_video.mp4", "wb") as f:
        f.write(uploaded_video.read())

    st.info("Analyzing video...")

    result = analyze_video("temp_video.mp4")

    if "error" in result:
        st.error(result["error"])
    else:
        st.subheader("Detection Result")
        st.write("Final Score:", result["score"])
        st.write("Label:", result["label"])
        st.write("Spatial Variance:", result["sharpness_var"])
        st.write("Temporal Variance:", result["motion_var"])
        st.write("Frequency Variance:", result["frequency_var"])

        if result["heatmap"] is not None:
            st.subheader("Explainability (Grad-CAM)")
            st.image(result["heatmap"], use_column_width=True)
