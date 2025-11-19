# posture_realtime_stable_text.py
import cv2
import mediapipe as mp
import numpy as np
import time

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def angle_between(v1, v2):
    v1 = np.array(v1, dtype=float)
    v2 = np.array(v2, dtype=float)
    dot = np.dot(v1, v2)
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    cos_theta = np.clip(dot / (n1 * n2), -1.0, 1.0)
    return np.degrees(np.arccos(cos_theta))

def classify_posture(landmarks, image_shape):
    h, w, _ = image_shape

    def get_point(idx):
        lm = landmarks[idx]
        return np.array([lm.x * w, lm.y * h])

    left_ear = get_point(mp_pose.PoseLandmark.LEFT_EAR.value)
    left_shoulder = get_point(mp_pose.PoseLandmark.LEFT_SHOULDER.value)
    left_hip = get_point(mp_pose.PoseLandmark.LEFT_HIP.value)

    neck_vec = left_ear - left_shoulder
    vertical_vec = np.array([0, -1])

    neck_angle = angle_between(neck_vec, vertical_vec)

    torso_vec = left_shoulder - left_hip
    torso_angle = angle_between(torso_vec, vertical_vec)

    bad_neck = neck_angle > 25
    bad_torso = torso_angle > 15
    posture_ok = not (bad_neck or bad_torso)

    score = 100
    score -= max(0, neck_angle - 10) * 1.5
    score -= max(0, torso_angle - 5) * 2.0
    score = int(np.clip(score, 0, 100))

    return posture_ok, neck_angle, torso_angle, score

def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # last stable values for text
    last_status_text = "NO DATA YET"
    last_color = (0, 255, 255)
    last_neck_angle = 0.0
    last_torso_angle = 0.0
    last_score = 0
    last_bad_duration = 0.0
    bad_start_time = time.time()
    last_update_time = 0.0            # when we last refreshed the text

    cv2.namedWindow("AI Posture Coach 2.0 (Lite)", cv2.WINDOW_NORMAL)

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=0,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)

            now = time.time()

            # only recompute + update text every 0.5s for stability
            if results.pose_landmarks and (now - last_update_time) > 0.5:
                landmarks = results.pose_landmarks.landmark
                posture_ok, neck_angle, torso_angle, score = classify_posture(
                    landmarks, frame.shape
                )

                if posture_ok:
                    last_color = (0, 255, 0)
                    last_status_text = "GOOD POSTURE"
                    bad_start_time = now
                    last_bad_duration = 0.0
                else:
                    last_color = (0, 0, 255)
                    last_bad_duration = now - bad_start_time
                    last_status_text = f"BAD POSTURE ({last_bad_duration:.1f}s)"

                last_neck_angle = neck_angle
                last_torso_angle = torso_angle
                last_score = score
                last_update_time = now

            # draw landmarks if present
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
                )

            # draw text using last_* values EVERY frame (no flicker)
            color = last_color
            cv2.putText(frame, last_status_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            cv2.putText(frame, f"Neck angle: {last_neck_angle:.1f} deg",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(frame, f"Torso angle: {last_torso_angle:.1f} deg",
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(frame, f"Posture score: {last_score}/100",
                        (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            cv2.imshow("AI Posture Coach 2.0 (Lite)", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()