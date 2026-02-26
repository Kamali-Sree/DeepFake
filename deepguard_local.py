import streamlit as st
import cv2
import numpy as np
import os
import torch
from facenet_pytorch import MTCNN

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
mtcnn = MTCNN(keep_all=False, device=DEVICE)

VIDEO_FOLDER = "videos"

def sample_frames(video_path, n=12):
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

def analyze_video(video_path):

    frames = sample_frames(video_path)

    sharpness_scores = []
    frame_diffs = []
    prev_gray = None

    for frame in frames:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face = mtcnn(rgb)

        if face is None:
            continue

        face_np = face.squeeze().permute(1,2,0).cpu().numpy()
        face_np = (face_np * 255).astype(np.uint8)
        gray = cv2.cvtColor(face_np, cv2.COLOR_RGB2GRAY)

        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        sharpness_scores.append(sharpness)

        if prev_gray is not None:
            diff = np.mean(np.abs(gray - prev_gray))
            frame_diffs.append(diff)

        prev_gray = gray

    if len(sharpness_scores) == 0:
        return {"error": "No face detected"}

    sharpness_var = np.var(sharpness_scores)
    motion_var = np.var(frame_diffs) if len(frame_diffs) > 0 else 0

    final_score = 0.6 * sharpness_var + 0.4 * motion_var
    final_score = min(final_score / 1000, 1.0)

    if final_score < 0.4:
        label = "Likely Authentic"
    elif final_score < 0.7:
        label = "Inconclusive"
    else:
        label = "High Deepfake Risk"

    return {
        "score": round(final_score, 3),
        "label": label,
        "sharpness_var": round(sharpness_var, 3),
        "motion_var": round(motion_var, 3)
    }

# ---------------- UI ----------------

st.title("🛡 DeepGuard – Server-Centric Detection")

query_params = st.query_params
video_name = query_params.get("video", [None])[0]

if video_name:
    st.info(f"Analyzing video: {video_name}")
    video_path = os.path.join(VIDEO_FOLDER, video_name)

    if os.path.exists(video_path):
        result = analyze_video(video_path)
        if "error" in result:
            st.error(result["error"])
        else:
            st.subheader("Detection Result")
            st.write("Score:", result["score"])
            st.write("Label:", result["label"])
            st.write("Sharpness Variance:", result["sharpness_var"])
            st.write("Motion Variance:", result["motion_var"])
    else:
        st.error("Video not found in videos folder.")

else:
    st.write("Upload or specify video parameter.")
    uploaded_video = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])
    if uploaded_video:
        os.makedirs(VIDEO_FOLDER, exist_ok=True)
        video_path = os.path.join(VIDEO_FOLDER, uploaded_video.name)
        with open(video_path, "wb") as f:
            f.write(uploaded_video.read())
        st.success("Uploaded. Reload with ?video=" + uploaded_video.name)
