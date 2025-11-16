import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path
import time
import simpleaudio as sa
from pathlib import Path
import time

ASSETS_DIR = Path(__file__).resolve().parent / "assets"

def load_ref_image(name):
    path = ASSETS_DIR / name
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)  # keep alpha if present
    if img is None:
        print(f"[WARN] Could not load {path}")
        return None

    # Handle grayscale
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    # Handle BGRA (with alpha)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    print(f"[INFO] Loaded {name} with shape {img.shape}, mean={img.mean():.1f}")
    return img


REF_IMAGES = {
    "Y": cv2.imread(str(ASSETS_DIR / "ASL_Y.png")),
    "M": cv2.imread(str(ASSETS_DIR / "ASL_M.png")),
    "C": cv2.imread(str(ASSETS_DIR / "ASL_C.png")),
    "A": cv2.imread(str(ASSETS_DIR / "ASL_A.png")),
}
for k, img in REF_IMAGES.items():
    print(k, "loaded:", img is not None, img.shape if img is not None else None)

TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"
AUDIO_PATH = Path(__file__).resolve().parent.parent / "src"/ "ymca-asl-tutor"/ "data" / "audio" / "ymca.wav"
AUDIO_PATH_BEGINNER = Path(__file__).resolve().parent.parent / "src"/ "ymca-asl-tutor"/ "data" / "audio" / "slow_ymca.wav"
mp_hands = mp.solutions.hands

TUTORIAL_ORDER = ["Y", "M", "C", "A"]
TUTORIAL_THRESHOLD = 0.3 #passing

performance_level = "normal"
SLOW_FACTOR = 1.6

# Fixed order and duration (seconds) for each letter in the YMCA chorus
CHORUS_TIMING = [
    ("Y", 3.0, 3.9),
    ("M", 3.9, 4.5),
    ("C", 4.5, 4.8),
    ("A", 4.8, 5.0),

    ("Y", 6.8, 7.6),
    ("M", 7.6, 8.0),
    ("C", 8.0, 8.3),
    ("A", 8.3, 9.0),
]
CHORUS_END = 9.1
ALPHA = 0.75  # bigger = more sensitive

CHORUS_TIMING_BEGINNER = [(label, start * SLOW_FACTOR, end * SLOW_FACTOR) for (label, start, end) in CHORUS_TIMING]
CHORUS_END_BEGINNER = CHORUS_END * SLOW_FACTOR


def extract_hand_vec(hand_landmarks):
    pts = np.array([(lm.x, lm.y) for lm in hand_landmarks.landmark], dtype=np.float32)
    origin = pts[0]
    pts = pts - origin
    scale = np.linalg.norm(pts[9]) + 1e-6
    pts = pts / scale
    return pts.flatten()


def load_templates():
    templates = {}
    for label in ["Y", "M", "C", "A"]:
        path = TEMPLATE_DIR / f"{label}.npy"
        if path.exists():
            templates[label] = np.load(path)
        else:
            print(f"[WARN] Template for {label} not found at {path}")
    return templates


def classify(vec, templates):
    best_label = None
    best_dist = float("inf")
    for label, tvec in templates.items():
        d = np.linalg.norm(vec - tvec)
        if d < best_dist:
            best_dist = d
            best_label = label
    # distance -> [0,1] score
    score = float(np.exp(-ALPHA * best_dist)) #HERE
    
    return best_label, score, best_dist


def get_expected_label(start_time, timing, chorus_end):
    """
    Given the wall-clock start_time of the audio,
    return (expected_label, progress_in_current_window, elapsed).
    """
    elapsed = time.time() - start_time
    t = elapsed % chorus_end

    for label, t_start, t_end in timing:
        if t_start <= t < t_end:
            duration = t_end - t_start
            progress = (t - t_start) / duration
            return label, progress, elapsed

    # Before first Y or after last A, no specific letter expected
    return None, 0.0, elapsed



def main():
    good_streak = 0
    GOOD_STREAK_TARGET = 10

    mode = "menu"   # "menu", "tutorial", "performance", "summary"
    summary_lines = []
    tutorial_index = 0  # index into TUTORIAL_ORDER

    templates = load_templates()
    if not templates:
        print("[ERROR] No templates found. Run capture_templates.py first.")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open webcam.")
        return

    per_letter_scores = {label: [] for label in templates.keys()}

    wave_obj = sa.WaveObject.from_wave_file(str(AUDIO_PATH))
    wave_beginner = sa.WaveObject.from_wave_file(str(AUDIO_PATH_BEGINNER))
    performance_level = "normal"
    current_wave = wave_obj
    current_timing = CHORUS_TIMING

    print("[INFO] Initial mode: MENU (press T for tutorial, S for performance, Q to quit)")

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
            h, w, _ = frame.shape

            # ---------- MENU MODE ----------
            if mode == "menu":
                # Simple dark overlay with menu text
                overlay = frame.copy()
                cv2.rectangle(overlay, (40, 40), (w - 40, h - 40), (0, 0, 0), -1)
                frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)

                cv2.putText(frame, "YMCA ASL Tutor", (60, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
                cv2.putText(frame, "T - Tutorial (learn letters)", (60, 180),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                cv2.putText(frame, "S - Start performance", (60, 220),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                cv2.putText(frame, "Q - Quit", (60, 260),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                cv2.putText(frame, "1 - Slow Mode", (60, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(frame, "2 - Normal Mode", (60, 340), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                cv2.imshow("YMCA ASL Tutor", frame)
                key = cv2.waitKey(1) & 0xFF

                if key in [ord("p"), ord("P"), ord("s"), ord("S")]:
                    mode = "difficulty"
                    print("[INFO] Choose difficulty: 1 = Slow, 2 = Normal")
                    # MODE_PERFORMANCE = "normal"
                    # print("[INFO] Performance mode set to NORMAL.")
                if key in [ord("q"), ord("Q")]:
                    break
                if key in [ord("t"), ord("T")]:
                    mode = "tutorial"
                    tutorial_index = 0
                    print("[INFO] Tutorial mode started (Y then M then C then A).")
                

            if mode == "difficulty":
                overlay = frame.copy()
                cv2.rectangle(overlay, (40, 40), (w - 40, h - 40), (0, 0, 0), -1)
                frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)

                cv2.putText(frame, "Select Difficulty", (60, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 3)
                cv2.putText(frame, "1 - Beginner (slower)", (60, 180),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                cv2.putText(frame, "2 - Normal", (60, 220),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                cv2.putText(frame, "M - Back to menu   Q - Quit", (60, h - 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                cv2.imshow("YMCA ASL Tutor", frame)
                key = cv2.waitKey(1) & 0xFF

                if key in [ord("q"), ord("Q")]:
                    break
                if key in [ord("m"), ord("M")]:
                    mode = "menu"
                    print("[INFO] Back to menu from difficulty.")
                if key == ord("1"):
                    performance_level = "beginner"
                    current_wave = wave_beginner
                    current_timing = CHORUS_TIMING_BEGINNER
                    current_chorus_end = CHORUS_END_BEGINNER
                    per_letter_scores = {label: [] for label in templates.keys()}
                    play_obj = current_wave.play()
                    start_time = time.time()
                    playing = True
                    mode = "performance"
                    print("[INFO] Beginner performance started.")
                if key == ord("2"):
                    performance_level = "normal"
                    current_wave = wave_obj
                    current_timing = CHORUS_TIMING
                    current_chorus_end = CHORUS_END
                    per_letter_scores = {label: [] for label in templates.keys()}
                    play_obj = current_wave.play()
                    start_time = time.time()
                    playing = True
                    mode = "performance"
                    print("[INFO] Normal performance started.")
                continue
            # ---------- SUMMARY MODE ----------
            if mode == "summary":
                overlay = frame.copy()
                cv2.rectangle(overlay, (40, 40), (w - 40, h - 40), (0, 0, 0), -1)
                frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)

                y0 = 80
                for i, line in enumerate(summary_lines):
                    y_line = y0 + i * 30
                    cv2.putText(frame, line, (60, y_line),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

                cv2.putText(frame, "R - Replay chorus", (60, h - 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(frame, "M - Back to menu   Q - Quit", (60, h - 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                cv2.imshow("YMCA ASL Tutor", frame)
                key = cv2.waitKey(1) & 0xFF

                if key in [ord("q"), ord("Q")]:
                    break
                if key in [ord("m"), ord("M")]:
                    mode = "menu"
                    print("[INFO] Back to menu.")
                if key in [ord("r"), ord("R")]:
                    per_letter_scores = {label: [] for label in templates.keys()}
                    play_obj = current_wave.play()
                    start_time = time.time()
                    playing = True
                    mode = "Performance"
                    print(f"[INFO] Chprus restarted in {performance_level} mode")
        

                    playing = True
                    mode = "performance"
                    print("[INFO] Chorus restarted from summary.")
                continue

            # ---------- TUTORIAL MODE ----------
            if mode == "tutorial":
                result = hands.process(rgb)
                current_label = TUTORIAL_ORDER[tutorial_index]
                predicted_label = "-"
                score = 0.0

                if result.multi_hand_landmarks:
                    hand_landmarks = result.multi_hand_landmarks[0]
                    # draw landmarks
                    for lm in hand_landmarks.landmark:
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        cv2.circle(frame, (cx, cy), 2, (0, 255, 0), -1)

                    vec = extract_hand_vec(hand_landmarks)
                    predicted_label, score, dist = classify(vec, templates)

                # UI overlays for tutorial
                cv2.putText(frame, f"Tutorial: {current_label}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                cv2.putText(frame, f"Predicted: {predicted_label}  Score: {score:.2f}",
                            (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                # Simple GOOD / KEEP TRYING indicator
                if score >= TUTORIAL_THRESHOLD and predicted_label == current_label:
                    msg = "GOOD!"
                    color = (0, 255, 0)
                    good_streak += 1
                else:
                    msg = "Keep trying..."
                    color = (0, 165, 255)
                    good_streak = max(0, good_streak - 1)
                
                progress = min(1, good_streak/  GOOD_STREAK_TARGET)

                #streak abr
                bar_x, bar_y, bar_w, bar_h = 10, 130, 200, 20
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (255, 255, 255), 2)
                filled_w = int(bar_w * progress)
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + filled_w, bar_y + bar_h), (0, 255, 0) if progress >= 1.0 else (0, 165, 255), -1)

                if progress >= 1.0:
                    msg = "Great, you've got it!"
                    color = (0, 255, 0)
                else:
                    msg = "Match the reference & hold"
                    color = (0, 165, 255)

                cv2.putText(frame, msg, (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

                ref_img = REF_IMAGES.get(current_label)
                if ref_img is not None:
                    ref_h = 200
                    scale = ref_h/ ref_img.shape[0]
                    ref_w = int(ref_img.shape[1] * scale)
                    ref_thumb = cv2.resize(ref_img, (ref_w, ref_h))
                    y_off = 40
                    x_off = w - ref_w - 20
                    frame[y_off:y_off + ref_h, x_off:x_off + ref_w] = ref_thumb

                    cv2.putText(frame, "Target Pose", (x_off, y_off - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 25, 255), 2)
                
                cv2.putText(frame, "N - Next letter   B - Previous",
                            (10, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (255, 255, 255), 2)
                cv2.putText(frame, "S - Start performance   M - Menu   Q - Quit",
                            (10, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (255, 255, 255), 2)
                
                cv2.imshow("YMCA ASL Tutor", frame)
                key = cv2.waitKey(1) & 0xFF

                if key in [ord("q"), ord("Q")]:
                    break
                if key in [ord("m"), ord("M")]:
                    mode = "menu"
                    print("[INFO] Back to menu from tutorial.")
                if key in [ord("n"), ord("N")]:
                    tutorial_index = (tutorial_index + 1) % len(TUTORIAL_ORDER)
                if key in [ord("b"), ord("B")]:
                    tutorial_index = (tutorial_index - 1) % len(TUTORIAL_ORDER)
                if key in [ord("p"), ord("P"), ord("s"), ord("S")]:
                    mode = "difficulty"
                    print("[INFO] Choose difficulty: 1 = Slow, 2 = Normal")
        

            # ---------- PERFORMANCE MODE ----------
            if mode == "performance":
                result = hands.process(rgb)

                if playing and start_time is not None:
                    expected_label, progress, elapsed = get_expected_label(start_time, current_timing, current_chorus_end)
                else:
                    expected_label, progress, elapsed = None, 0.0, 0.0

                predicted_label = "-"
                score = 0.0

                if result.multi_hand_landmarks and expected_label is not None:
                    hand_landmarks = result.multi_hand_landmarks[0]
                    vec = extract_hand_vec(hand_landmarks)
                    predicted_label, score, dist = classify(vec, templates)

                    if predicted_label == expected_label:
                        per_letter_scores[expected_label].append(score)

                # Overlays
                exp_text = expected_label if expected_label is not None else "-"
                cv2.putText(
                    frame,
                    f"Expected: {exp_text}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 255),
                    2,
                )
                cv2.putText(
                    frame,
                    f"Predicted: {predicted_label}  Score: {score:.2f}",
                    (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                )

                # progress bar for letter window
                bar_x, bar_y, bar_w, bar_h = 10, 90, 200, 20
                cv2.rectangle(
                    frame,
                    (bar_x, bar_y),
                    (bar_x + bar_w, bar_y + bar_h),
                    (255, 255, 255),
                    2,
                )
                filled_w = int(bar_w * progress)
                cv2.rectangle(
                    frame,
                    (bar_x, bar_y),
                    (bar_x + filled_w, bar_y + bar_h),
                    (0, 255, 255),
                    -1,
                )

                cv2.putText(
                    frame,
                    "S - Start (if not playing)   Q - Quit",
                    (10, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 200, 255),
                    2,
                )

                cv2.imshow("YMCA ASL Tutor", frame)
                key = cv2.waitKey(1) & 0xFF

                # start audio + timing if not already playing
                if not playing and key in [ord("s"), ord("S")]:
                    per_letter_scores = {label: [] for label in templates.keys()}
                    if performance_level == "beginner":
                        play_obj = wave_beginner.play()
                        start_time = time.time()
                        current_timing = CHORUS_TIMING_BEGINNER
                        current_chorus_end = CHORUS_END_BEGINNER
                    else:
                        play_obj = wave_obj.play()
                        start_time = time.time()
                        current_timing = CHORUS_TIMING
                        current_chorus_end = CHORUS_END
                    # play_obj = wave_obj.play()
                    # start_time = time.time()
                    playing = True
                    print("[INFO] Chorus started")

                if key in [ord("q"), ord("Q")]:
                    break

                # chorus finished -> go to summary
                if playing and elapsed >= current_chorus_end:
                    playing = False
                    mode = "summary"

                    summary_lines = []
                    summary_lines.append("YMCA ASL Performance Summary")
                    for label, scores in per_letter_scores.items():
                        if scores:
                            avg = float(np.mean(scores))
                            if avg >= 0.7:
                                fb = "You're doing great!"
                            elif avg >= 0.4:
                                fb = "Getting there!"
                            else:
                                fb = "Needs practice."
                            summary_lines.append(
                                f"{label}: {len(scores)} hits, avg {avg:.2f}, {fb}"
                            )
                        else:
                            summary_lines.append(
                                f"{label}: no confident matches"
                            )

                    print("[INFO] Chorus finished, showing summary.")

                continue  # end performance mode loop

    cap.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    main()
