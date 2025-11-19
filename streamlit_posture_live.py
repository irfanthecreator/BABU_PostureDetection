import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import joblib
import time
import threading
import os

# winsound is Windows-only; make it optional so the app works on Linux (Streamlit Cloud)
try:
    import winsound
    HAS_WINSOUND = True
except ImportError:
    winsound = None
    HAS_WINSOUND = False


# ================= SETUP =================
MODEL_PATH = "posture_model_best.pkl"  # or "posture_model.pkl"

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

BAD_THRESHOLD = 5.0  # seconds of continuous bad posture before beep


@st.cache_resource
def load_model(path: str):
    """Load the trained model once and cache it."""
    if not os.path.exists(path):
        st.error(
            f"Model file '{path}' not found. "
            f"Make sure it is in the app folder on GitHub."
        )
        return None
    return joblib.load(path)


clf = load_model(MODEL_PATH)


def beep_alert():
    """Short beep in a separate thread so it doesn't freeze video.
    On non-Windows (no winsound), this is a no-op.
    """
    if not HAS_WINSOUND:
        # On Streamlit Cloud (Linux), skip sound
        return

    def _beep():
        winsound.Beep(1000, 300)  # 1000 Hz, 0.3 sec

    threading.Thread(target=_beep, daemon=True).start()


def landmarks_to_feature_vector(landmarks):
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


def predict_posture(frame, pose):
    """Run Mediapipe + model prediction on a frame."""
    # if model didn't load, don't crash
    if clf is None:
        return frame, None, 0.0

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(rgb)

    if not result.pose_landmarks:
        return frame, None, 0.0

    # draw skeleton
    mp_drawing.draw_landmarks(
        frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS
    )

    feats = np.array(
        landmarks_to_feature_vector(result.pose_landmarks.landmark)
    ).reshape(1, -1)

    probs = clf.predict_proba(feats)[0]
    classes = clf.classes_
    idx = probs.argmax()
    label = classes[idx]
    conf = probs[idx]

    return frame, label, conf


# ================= UI =================
st.set_page_config(page_title="Live Posture Detection", page_icon="🟢")
st.title("🟢 REAL-TIME AI Posture Detection (Streamlit Live)")
st.write(
    f"Model: `{MODEL_PATH}`  •  Beeps after {BAD_THRESHOLD:.0f} seconds of continuous bad posture "
    f"(sound only works on Windows)."
)

start_btn = st.button("Start Webcam")

FRAME_WINDOW = st.empty()
info_box = st.empty()
timer_box = st.empty()

if start_btn:
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        info_box.error("Cannot access webcam. "
                       "On Streamlit Cloud, the server cannot see your local camera.")
    else:
        bad_start_time = None
        bad_beeped = False

        with mp_pose.Pose(
            static_image_mode=False,
            model_complexity=0,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as pose:

            while True:
                ret, frame = cap.read()
                if not ret:
                    info_box.error("Cannot read from webcam.")
                    break

                # mirror like a mirror
                frame = cv2.flip(frame, 1)

                frame, label, conf = predict_posture(frame, pose)
                now = time.time()

                # ----- bad posture timing logic -----
                bad_duration = 0.0
                if label == "bad":
                    if bad_start_time is None:
                        bad_start_time = now  # start timing
                    bad_duration = now - bad_start_time

                    # beep once when threshold crossed
                    if bad_duration >= BAD_THRESHOLD and not bad_beeped:
                        beep_alert()
                        bad_beeped = True

                    # draw timer text on frame
                    cv2.putText(
                        frame,
                        f"BAD for {bad_duration:.1f}s",
                        (10, 65),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 255),
                        2,
                    )
                else:
                    # reset timer if posture is good or no label
                    bad_start_time = None
                    bad_beeped = False

                # ----- main label text -----
                if label is None:
                    info_box.warning("No pose detected or model not loaded...")
                    timer_box.empty()
                else:
                    color = (0, 255, 0) if label == "good" else (0, 0, 255)
                    cv2.putText(
                        frame,
                        f"{label.upper()} ({conf*100:.1f}%)",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        color,
                        2,
                    )

                    info_box.success(f"Posture: {label.upper()}  ({conf*100:.1f}%)")
                    if label == "bad":
                        timer_box.warning(
                            f"BAD posture for {bad_duration:.1f} seconds "
                            f"(beep at {BAD_THRESHOLD:.0f}s)"
                        )
                    else:
                        timer_box.info("Posture is GOOD ✅")

                # show frame in Streamlit
                FRAME_WINDOW.image(
                    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                    channels="RGB",
                )

                # safety break to avoid infinite loop if cap suddenly closes
                if not cap.isOpened():
                    break

        cap.release()
