import time
import threading
import os

import cv2
import mediapipe as mp
import numpy as np
import joblib
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode, RTCConfiguration

# winsound is Windows-only; make it optional so the app works on Linux (Streamlit Cloud)
try:
    import winsound
    HAS_WINSOUND = True
except ImportError:
    winsound = None
    HAS_WINSOUND = False


# ====================== CONFIG ======================
MODEL_PATH = "posture_model_best.pkl"  # or "posture_model.pkl"
BAD_THRESHOLD = 5.0  # seconds of continuous bad posture before beep

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

RTC_CONFIGURATION = {
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
}


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
    """
    Short beep in a separate thread.
    On non-Windows (no winsound), this does nothing.
    """
    if not HAS_WINSOUND:
        return

    def _beep():
        winsound.Beep(1000, 300)  # 1000 Hz, 0.3 sec

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


# ================== VIDEO PROCESSOR ==================
class PostureVideoProcessor(VideoTransformerBase):
    def __init__(self):
        # mediapipe pose instance (reuse for all frames)
        self.pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=0,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        # timing for bad posture
        self.bad_start_time = None
        self.bad_beeped = False
        self.last_label = None
        self.last_conf = 0.0
        self.last_bad_duration = 0.0

    def recv(self, frame):
        # frame from browser → numpy BGR
        img = frame.to_ndarray(format="bgr24")

        # mirror horizontally (like webcam)
        img = cv2.flip(img, 1)

        # if model failed to load, just return the image
        if clf is None:
            cv2.putText(
                img,
                "Model not loaded",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                2,
            )
            from av import VideoFrame
            return VideoFrame.from_ndarray(img, format="bgr24")

        # run mediapipe
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = self.pose.process(rgb)

        label = None
        conf = 0.0
        now = time.time()
        bad_duration = 0.0

        if result.pose_landmarks:
            # draw skeleton
            mp_drawing.draw_landmarks(
                img, result.pose_landmarks, mp_pose.POSE_CONNECTIONS
            )

            feats = np.array(
                landmarks_to_feature_vector(result.pose_landmarks.landmark)
            ).reshape(1, -1)

            probs = clf.predict_proba(feats)[0]
            classes = clf.classes_
            idx = probs.argmax()
            label = classes[idx]
            conf = probs[idx]

            # ===== bad posture timing logic =====
            if label == "bad":
                if self.bad_start_time is None:
                    self.bad_start_time = now
                bad_duration = now - self.bad_start_time

                if bad_duration >= BAD_THRESHOLD and not self.bad_beeped:
                    beep_alert()
                    self.bad_beeped = True

                # draw timer
                cv2.putText(
                    img,
                    f"BAD for {bad_duration:.1f}s",
                    (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                )
            else:
                self.bad_start_time = None
                self.bad_beeped = False

            # ===== main label text =====
            color = (0, 255, 0) if label == "good" else (0, 0, 255)
            cv2.putText(
                img,
                f"{label.upper()} ({conf*100:.1f}%)",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                color,
                2,
            )
        else:
            # no pose detected
            cv2.putText(
                img,
                "No pose detected",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                2,
            )

        # store last info so UI can read it if needed
        self.last_label = label
        self.last_conf = conf
        self.last_bad_duration = bad_duration

        from av import VideoFrame
        return VideoFrame.from_ndarray(img, format="bgr24")


# ====================== STREAMLIT UI ======================
st.set_page_config(page_title="Live Posture Detection (WebRTC)", page_icon="🟢")

st.title("🟢 REAL-TIME AI Posture Detection (WebRTC)")
st.write(
    f"Model: `{MODEL_PATH}` • Beeps after {BAD_THRESHOLD:.0f} seconds of continuous bad posture "
    f"(sound only works on Windows, and only if running locally)."
)
st.write(
    "➡ Allow camera access in your browser.\n"
    "Frames from your webcam are sent to the server, where Mediapipe + your model "
    "run in real-time, then the processed video is streamed back."
)

webrtc_ctx = webrtc_streamer(
    key="posture-live",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
    video_processor_factory=PostureVideoProcessor,
    async_processing=True,
)

# Optional: show live status text under the video
status_placeholder = st.empty()

if webrtc_ctx and webrtc_ctx.video_processor:
    vp = webrtc_ctx.video_processor
    # this block will rerun repeatedly as Streamlit reruns the script
    label = vp.last_label
    conf = vp.last_conf
    bad_duration = vp.last_bad_duration

    if label is None:
        status_placeholder.warning("No pose detected yet…")
    else:
        if label == "good":
            status_placeholder.success(
                f"Posture: GOOD ({conf*100:.1f}%)."
            )
        else:
            status_placeholder.warning(
                f"Posture: BAD ({conf*100:.1f}%). "
                f"Bad for {bad_duration:.1f}s (beep at {BAD_THRESHOLD:.0f}s)."
            )
else:
    status_placeholder.info("Click on the video box and allow camera access to start.")
