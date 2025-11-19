import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
import joblib
import winsound
import threading

# ==== CONFIG ====
MODEL_PATH = "posture_model_best.pkl"  # or "posture_model.pkl"

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# ---- load trained model (cached) ----
@st.cache_resource
def load_model(path):
    return joblib.load(path)

clf = load_model(MODEL_PATH)


def beep_alert():
    """short beep in a thread so the UI doesn't freeze"""
    def _beep():
        winsound.Beep(1000, 300)  # 1000 Hz, 0.3s
    threading.Thread(target=_beep, daemon=True).start()


def landmarks_to_feature_vector(landmarks):
    """
    Same feature extraction as training:
    NOSE, L_EAR, R_EAR, L_SHOULDER, R_SHOULDER, L_HIP, R_HIP
    -> 7 joints * (x,y,z) = 21 features
    """
    key_ids = [
        mp_pose.PoseLandmark.NOSE,
        mp_pose.PoseLandmark.LEFT_EAR,
        mp_pose.PoseLandmark.RIGHT_EAR,
        mp_pose.PoseLandmark.LEFT_SHOULDER,
        mp_pose.PoseLandmark.RIGHT_SHOULDER,
        mp_pose.PoseLandmark.LEFT_HIP,
        mp_pose.PoseLandmark.RIGHT_HIP,
    ]
    features = []
    for lm_id in key_ids:
        lm = landmarks[lm_id.value]
        features.extend([lm.x, lm.y, lm.z])
    return features


def analyze_image(pil_img):
    """
    Run Mediapipe on the image, classify posture, and return:
    - output image with skeleton + text
    - predicted label
    - confidence
    """
    img = np.array(pil_img.convert("RGB"))  # RGB
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:
        results = pose.process(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

        if not results.pose_landmarks:
            return img, None, None

        # draw skeleton on image
        mp_drawing.draw_landmarks(
            img_bgr, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
        )

        # extract features
        feats = np.array(
            landmarks_to_feature_vector(results.pose_landmarks.landmark)
        ).reshape(1, -1)

        # predict with trained model
        probs = clf.predict_proba(feats)[0]
        classes = clf.classes_
        best_idx = probs.argmax()
        pred_label = classes[best_idx]
        pred_conf = probs[best_idx]

        # overlay text
        color = (0, 255, 0) if pred_label == "good" else (0, 0, 255)
        text = f"{pred_label.upper()} ({pred_conf*100:.1f}%)"

        cv2.putText(
            img_bgr,
            text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            color,
            2,
        )

        # convert back to RGB for display
        out_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return out_img, pred_label, pred_conf


# ================== STREAMLIT UI ==================

st.set_page_config(page_title="AI Posture Test App", page_icon="🧍", layout="centered")

st.title("🧍 AI Posture Classifier (Streamlit Test)")
st.write(
    "Upload a photo or use the camera to test the trained posture model. "
    "If posture is **BAD**, a beep will play."
)

tab1, tab2 = st.tabs(["📁 Upload Image", "📷 Use Camera"])

# ---------- TAB 1: UPLOAD IMAGE ----------
with tab1:
    uploaded_file = st.file_uploader(
        "Upload an image (JPG/PNG)", type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        pil_img = Image.open(uploaded_file)
        st.image(pil_img, caption="Original Image", use_column_width=True)

        if st.button("Analyze Uploaded Image", key="analyze_upload"):
            with st.spinner("Analyzing posture..."):
                out_img, label, conf = analyze_image(pil_img)

            if label is None:
                st.warning("No pose detected. Try another image or clearer view.")
            else:
                st.image(out_img, caption="Pose & Prediction", use_column_width=True)
                st.success(f"Prediction: **{label.upper()}** ({conf*100:.1f}%)")
                if label == "bad":
                    beep_alert()
                    st.error("Detected BAD posture – please correct your posture.")

# ---------- TAB 2: CAMERA ----------
with tab2:
    st.write("Use your webcam to capture a test photo for posture analysis.")
    camera_img = st.camera_input("Take a picture")

    if camera_img is not None:
        pil_cam_img = Image.open(camera_img)
        st.image(pil_cam_img, caption="Captured Image", use_column_width=True)

        if st.button("Analyze Camera Image", key="analyze_camera"):
            with st.spinner("Analyzing posture..."):
                out_img, label, conf = analyze_image(pil_cam_img)

            if label is None:
                st.warning("No pose detected. Try sitting fully in frame, side view.")
            else:
                st.image(out_img, caption="Pose & Prediction", use_column_width=True)
                st.success(f"Prediction: **{label.upper()}** ({conf*100:.1f}%)")
                if label == "bad":
                    beep_alert()
                    st.error("Detected BAD posture – please correct your posture.")
