import cv2
import mediapipe as mp
import numpy as np
import csv
import os

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

OUTPUT_CSV = "posture_data.csv"

# labels you want to record
# you can add more later if you want
LABELS = {
    "1": "good",   # sitting upright, good neck/back
    "2": "bad"     # slouching / head forward / leaning badly
    # "3": "lean_left",
    # "4": "lean_right",
}

def landmarks_to_feature_vector(landmarks):
    """
    Convert a subset of pose landmarks into a 1D feature vector.
    Using a few key joints to keep it simple and light.
    """
    # important joints for posture
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
        # normalized coordinates (0–1), z is depth-ish
        features.extend([lm.x, lm.y, lm.z])

    return features  # length 7 joints * 3 = 21 features


def init_csv(path):
    """Create CSV file with header if it doesn't exist."""
    if not os.path.exists(path):
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            header = [f"f{i}" for i in range(21)] + ["label"]
            writer.writerow(header)
        print(f"[INFO] created new dataset file: {path}")
    else:
        print(f"[INFO] appending to existing dataset file: {path}")


def main():
    init_csv(OUTPUT_CSV)

    cap = cv2.VideoCapture(0)
    cv2.namedWindow("Collect posture data", cv2.WINDOW_NORMAL)

    print("press:")
    print("  1 = GOOD posture")
    print("  2 = BAD posture")
    print("  q = quit")

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

            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
                )

            cv2.putText(
                frame,
                "1=good, 2=bad, q=quit",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
            )

            cv2.imshow("Collect posture data", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            # if we have pose and user pressed a label key
            if results.pose_landmarks and chr(key) in LABELS:
                label = LABELS[chr(key)]
                feats = landmarks_to_feature_vector(results.pose_landmarks.landmark)
                row = feats + [label]

                with open(OUTPUT_CSV, "a", newline="") as f:
                    csv.writer(f).writerow(row)

                print(f"[SAVED] one sample as {label}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()