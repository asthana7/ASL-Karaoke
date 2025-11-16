# capture_templates.py

import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path
import time

TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"
TEMPLATE_DIR.mkdir(exist_ok=True, parents=True)

mp_hands = mp.solutions.hands

# How many frames to average per capture
FRAMES_PER_CAPTURE = 15


def extract_hand_vec(hand_landmarks):
    """
    Convert 21 MediaPipe hand landmarks into a normalized 2D vector.
    Translation-invariant and approximately scale-invariant.
    """
    pts = np.array([(lm.x, lm.y) for lm in hand_landmarks.landmark], dtype=np.float32)
    # Wrist as origin
    origin = pts[0]
    pts = pts - origin
    # Scale by distance wrist -> middle finger MCP (landmark 9) to normalize size
    scale = np.linalg.norm(pts[9]) + 1e-6
    pts = pts / scale
    return pts.flatten()  # shape (42,)


def save_template(label, vectors):
    # If a previous template exists, load and concatenate
    out_path = TEMPLATE_DIR / f"{label}.npy"
    new_arr = np.stack(vectors, axis=0)

    if out_path.exists():
        old_arr = np.load(out_path)
        all_vecs = np.vstack([old_arr, new_arr])
    else:
        all_vecs = new_arr

    mean_vec = all_vecs.mean(axis=0)
    np.save(out_path, mean_vec)
    print(f"[INFO] Updated template for {label} with {all_vecs.shape[0]} samples at {out_path}")



def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open webcam.")
        return

    current_label = None
    capture_count = 0
    collected_vecs = []

    print("[INFO] Press Y / M / C / A to capture that letter template.")
    print("[INFO] Press Q to quit.")

    with mp_hands.Hands(
        model_complexity=1,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as hands:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            h, w, _ = frame.shape

            # Draw landmarks for feedback
            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    for lm in hand_landmarks.landmark:
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        cv2.circle(frame, (cx, cy), 2, (0, 255, 0), -1)

            # Text overlays
            msg1 = "Press Y / M / C / A to start capture (15 frames)."
            msg2 = f"Current label: {current_label or '-'} | Frames captured: {capture_count}"
            cv2.putText(frame, msg1, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, msg2, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            cv2.imshow("Capture Templates", frame)
            key = cv2.waitKey(1) & 0xFF

            if key in [ord("q"), ord("Q")]:
                break

            # Start capture for a given label
            if key in [ord("y"), ord("Y"), ord("m"), ord("M"), ord("c"), ord("C"), ord("a"), ord("A")]:
                current_label = chr(key).upper()
                capture_count = 0
                collected_vecs = []
                print(f"[INFO] Starting capture for {current_label}...")

            # If we're in capture mode and see a hand, record vectors
            if current_label and result.multi_hand_landmarks:
                hand_landmarks = result.multi_hand_landmarks[0]
                vec = extract_hand_vec(hand_landmarks)
                collected_vecs.append(vec)
                capture_count += 1

                if capture_count >= FRAMES_PER_CAPTURE:
                    save_template(current_label, collected_vecs)
                    current_label = None
                    capture_count = 0
                    collected_vecs = []

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
