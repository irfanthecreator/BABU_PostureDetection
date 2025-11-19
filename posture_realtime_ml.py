import cv2
import mediapipe as mp
import numpy as np
import time
import joblib

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

MODEL_PATH = "posture_model_best.pkl"

# load trained model
clf = joblib.load(MODEL_PATH)

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

def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    cv2.namedWindow("AI Posture Coach (Trained)", cv2.WINDOW_NORMAL)

    last_label = "NO DATA"
    last_conf = 0.0
    last_update_time = 0.0
    bad_start_time = time.time()
    bad_duration = 0.0
    last_color = (0, 255, 255)

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
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)
            now = time.time()

            if results.pose_landmarks and (now - last_update_time) > 0.3:
                feats = np.array(landmarks_to_feature_vector(results.pose_landmarks.landmark)).reshape(1, -1)
                probs = clf.predict_proba(feats)[0]
                classes = clf.classes_
                best_idx = probs.argmax()
                last_label = classes[best_idx]
                last_conf = probs[best_idx]

                if last_label == "bad":
                    last_color = (0, 0, 255)
                    bad_duration = now - bad_start_time
                else:
                    last_color = (0, 255, 0)
                    bad_start_time = now
                    bad_duration = 0.0

                last_update_time = now

            # draw skeleton
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
                )

            # overlay text (stable)
            color = last_color
            status_text = f"{last_label.upper()} ({last_conf*100:.1f}%)"
            cv2.putText(frame, status_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            if last_label == "bad":
                cv2.putText(frame, f"BAD for {bad_duration:.1f}s",
                            (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            cv2.imshow("AI Posture Coach (Trained)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
