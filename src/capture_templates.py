import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path
import time

# -------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
TEMPLATE_DIR = BASE_DIR / "templates"
TEMPLATE_DIR.mkdir(exist_ok=True, parents=True)

# -------------------------------------------------------------------
# MediaPipe setup
# -------------------------------------------------------------------
mp_hands = mp.solutions.hands

# How many frames to average per capture
FRAMES_PER_CAPTURE = 15
CAPTURE_DELAY = 1.0
LABEL_SETS = {
    "YMCA": {
        "y": "Y",
        "m": "M",
        "c": "C",
        "a": "A",
    },
    "LYLLS": {
        "b":"BABY",
        "i": "ILOVEYOU",   
        "k": "LIKE", 
        "a": "A", 
        "l": "LOVE",  
         "s":"SONG" 
    },
}

# -------------------------------------------------------------------
# Feature extraction helpers
# -------------------------------------------------------------------
def extract_hand_vec(hand_landmarks):
    """
    Convert 21 MediaPipe hand landmarks into a normalized 2D feature vector.
    Translation + (approx) scale invariant.
    Returns a 42-D vector (21 points * 2 coords).
    """
    pts = np.array([(lm.x, lm.y) for lm in hand_landmarks.landmark], dtype=np.float32)
    origin = pts[0]              # wrist as origin
    pts = pts - origin
    scale = np.linalg.norm(pts[9]) + 1e-6   # distance to middle-finger MCP
    pts = pts / scale
    return pts.flatten()         # shape (42,)


def extract_norm_coords(hand_landmarks):
    """
    Extract normalized (x, y, z) coords for ghost-hand overlay.

    - Center on wrist (landmark 0) -> translation invariance
    - Scale so max distance from origin is 1 -> scale invariance
    """
    coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark], dtype=np.float32)
    # center at wrist
    coords -= coords[0]
    # scale to unit size
    max_dist = np.max(np.linalg.norm(coords[:, :2], axis=1)) + 1e-6
    coords /= max_dist
    return coords  # shape (21,3)


# -------------------------------------------------------------------
# Saving templates
# -------------------------------------------------------------------
def save_template(label, vectors, coords_list):
    """
    Save / update:
      - feature vector template: <label>.npy
      - ghost-hand coords template: <label>_coords.npy
    """
    # --- Feature vectors ---
    feature_path = TEMPLATE_DIR / f"{label}.npy"
    new_vecs = np.stack(vectors, axis=0)  # (N, 42)

    if feature_path.exists():
        old_vecs = np.load(feature_path)
        if old_vecs.ndim == 1:
            old_vecs = old_vecs[None, :]
        all_vecs = np.vstack([old_vecs, new_vecs])
    else:
        all_vecs = new_vecs

    mean_vec = all_vecs.mean(axis=0)
    np.save(feature_path, mean_vec)
    print(f"[INFO] Updated feature template for {label} with {all_vecs.shape[0]} samples at {feature_path}")

    # --- Ghost-hand coords ---
    ghost_path = TEMPLATE_DIR / f"{label}_coords.npy"
    coords_arr = np.stack(coords_list, axis=0)  # (N, 21, 3)
    mean_coords = coords_arr.mean(axis=0)       # (21, 3)
    np.save(ghost_path, mean_coords)
    print(f"[INFO] Saved ghost coords for {label} at {ghost_path}")


# -------------------------------------------------------------------
# Main capture loop
# -------------------------------------------------------------------
def main():
    # --- Choose template set ---
    print("Select template set to capture:")
    print("1 - YMCA (letters: Y, M, C, A)")
    print("2 - LYLLS (words/signs: BABY, ILY, LIKE, A, LOVE, SONG)")
    choice = input("Enter 1 or 2: ").strip()

    if choice == "1":
        mode_name = "YMCA"
    elif choice == "2":
        mode_name = "LYLLS"
    else:
        print("Invalid choice, defaulting to YMCA.")
        mode_name = "YMCA"

    VALID_KEYS = LABEL_SETS[mode_name]
    print(f"[INFO] Capturing templates for mode: {mode_name}")
    print("[INFO] Keys:")
    for k, v in VALID_KEYS.items():
        print(f"   {k.upper()} → {v}")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open webcam.")
        return

    print("[INFO] Template directory:", TEMPLATE_DIR)
    print("[INFO] Press Y / M / C / A to start capturing that letter.")
    print(f"[INFO] Each capture averages {FRAMES_PER_CAPTURE} frames.")
    print("[INFO] Press Q to quit.")

    current_label = None
    frame_count = 0
    vec_buffer = []
    coords_buffer = []
    capture_ready_time = None
    capture_active = False

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

            # Draw landmarks if present
            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    for lm in hand_landmarks.landmark:
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        cv2.circle(frame, (cx, cy), 2, (0, 255, 0), -1)

            # Overlay text
            status_line = f"Current label: {current_label or '-'} | Frames captured: {frame_count}"

            if choice == 2:
                cv2.putText(frame, "Press B / I / K / A / L / S to capture template",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            else:
                cv2.putText(frame, "Press Y / M / C / A to capture template",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, status_line,
                        (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(frame, "Q - Quit",
                        (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

            cv2.imshow("Capture ASL Templates", frame)
            key = cv2.waitKey(1) & 0xFF

            # Quit
            if key in [ord("q"), ord("Q")]:
                break

            # Start capture for a given label (based on chosen mode)
            ch = chr(key).lower()
            if ch in VALID_KEYS:
                current_label = VALID_KEYS[ch]
                # set a future time when capture will actually start
                capture_ready_time = time.time() + CAPTURE_DELAY
                capture_active = False
                frame_count = 0
                vec_buffer = []
                coords_buffer = []
                print(f"[INFO] Get ready for {current_label} – capture starts in {CAPTURE_DELAY:.1f} seconds.")

            # If in capture mode and we see a hand, record frames
            now = time.time()

            # If we have a label armed but capture hasn't started yet, wait for delay
            if current_label and capture_ready_time is not None and not capture_active:
                remaining = capture_ready_time - now
                if remaining <= 0:
                    capture_active = True
                    frame_count = 0
                    vec_buffer = []
                    coords_buffer = []
                    print(f"[INFO] Capturing {current_label} NOW... hold the sign steady.")
                else:
                    # Optional: show countdown on screen
                    cv2.putText(frame, f"Starting {current_label} in {remaining:.1f}s",
                                (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                (0, 255, 255), 2)

            # If capture is active, actually record frames
            if current_label and capture_active and result.multi_hand_landmarks:
                hand_landmarks = result.multi_hand_landmarks[0]
                vec = extract_hand_vec(hand_landmarks)
                coords = extract_norm_coords(hand_landmarks)

                vec_buffer.append(vec)
                coords_buffer.append(coords)
                frame_count += 1

                cv2.putText(frame, f"Capturing {current_label}: {frame_count}/{FRAMES_PER_CAPTURE}",
                            (10, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 255, 0), 2)

                if frame_count >= FRAMES_PER_CAPTURE:
                    save_template(current_label, vec_buffer, coords_buffer)
                    print("[INFO] Capture complete. Press another key or Q to quit.")
                    current_label = None
                    capture_ready_time = None
                    capture_active = False
                    frame_count = 0
                    vec_buffer = []
                    coords_buffer = []

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Template capture finished.")


if __name__ == "__main__":
    main()
