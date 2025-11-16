import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path
import time
import simpleaudio as sa



YMCA_LABELS = ["Y", "M", "C", "A"]
LYLLS_LABELS = ["BABY", "ILOVEYOU", "LIKE", "A", "LOVE", "SONG"]

TUTORIAL_ORDER_YMCA = YMCA_LABELS
TUTORIAL_ORDER_LYLLS = LYLLS_LABELS


mp_hands = mp.solutions.hands
HAND_CONNECTIONS = mp_hands.HAND_CONNECTIONS
BASE_DIR = Path(__file__).resolve().parent
TEMPLATE_DIR = BASE_DIR / "templates"

print("[DEBUG] BASE_DIR:", BASE_DIR)
print("[DEBUG] TEMPLATE_DIR:", TEMPLATE_DIR, "exists:", TEMPLATE_DIR.exists())


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

    "BABY": cv2.imread(str(ASSETS_DIR / "ASL_BABY.jpg")),
    "ILOVEYOU": cv2.imread(str(ASSETS_DIR / "ASL_ILY.jpg")),
    "LIKE": cv2.imread(str(ASSETS_DIR / "ASL_LIKE.jpg")),
    "A": cv2.imread(str(ASSETS_DIR / "ASL_A.png")),
    "LOVE": cv2.imread(str(ASSETS_DIR / "ASL_LOVE.jpg")),
    "SONG": cv2.imread(str(ASSETS_DIR / "ASL_SONG.jpg")),
}
for k, img in REF_IMAGES.items():
    print(k, "loaded:", img is not None, img.shape if img is not None else None)



TEMPLATE_DIR = BASE_DIR/ "templates"
def load_templates_for_song(song_name):
    cfg = SONG_CONFIGS[song_name]
    required_labels = cfg["labels"]

    templates = {}
    ghost_coords = {}
    missing = []

    for label in required_labels:
        feat_path = TEMPLATE_DIR / f"{label}.npy"
        coord_path = TEMPLATE_DIR / f"{label}_coords.npy"

        if not feat_path.exists() or not coord_path.exists():
            missing.append(label)
            continue

        templates[label] = np.load(feat_path)
        ghost_coords[label] = np.load(coord_path)

    return templates, ghost_coords, missing
    


BASE_DIR = Path(__file__).resolve().parent

AUDIO_PATH_YMCA_NORMAL = BASE_DIR / "ymca-asl-tutor" / "data" / "audio" / "ymca.wav"
AUDIO_PATH_YMCA_SLOW   = BASE_DIR / "ymca-asl-tutor" / "data" / "audio" /"slow_ymca.wav"

AUDIO_PATH_LYLLS_NORMAL = BASE_DIR / "ymca-asl-tutor" / "data" / "audio" / "lylls.wav"
AUDIO_PATH_LYLLS_SLOW   = BASE_DIR / "ymca-asl-tutor" / "data" / "audio" / "slow_lylls.wav"  # optional, can point to same as normal if you don’t slow it



# TUTORIAL_ORDER_YMCA = ["Y", "M", "C", "A"]
# TUTORIAL_ORDER_LYLLS = ["B", "I", "K", "A", "L", "S"]
TUTORIAL_THRESHOLD = 0.25 #passing


performance_level = "normal"
SLOW_FACTOR = 1.6

# Fixed order and duration (seconds) for each letter in the YMCA chorus
CHORUS_TIMING_YMCA = [
    ("Y", 3.0, 3.9),
    ("M", 3.9, 4.5),
    ("C", 4.5, 4.8),
    ("A", 4.8, 5.0),

    ("Y", 6.8, 7.6),
    ("M", 7.6, 8.0),
    ("C", 8.0, 8.3),
    ("A", 8.3, 9.0),
]

CHORUS_TIMING_LYLLS = [
    ("BABY", 2.9, 4.0),
    ("ILOVEYOU", 4.0, 5.3), 
    ("LIKE", 5.3, 5.7), 
    ("A", 5.7, 5.9), 
    ("LOVE", 5.9, 6.4), 
    ("SONG", 6.4, 6.9), 

    ("BABY", 6.9, 7.8), 
    ("ILOVEYOU", 7.8, 9.6), 
    ("LIKE", 9.6, 10.0), 
    ("A", 10.0, 10.2), 
    ("LOVE", 10.2, 10.7), 
    ("SONG", 10.7, 11.1)
]
CHORUS_END_YMCA = 9.1
CHORUS_END_LYLLS = 11.1


ALPHA = 0.75  # bigger = more sensitive

CHORUS_TIMING_BEGINNER_YMCA = [(label, start * SLOW_FACTOR, end * SLOW_FACTOR) for (label, start, end) in CHORUS_TIMING_YMCA]
CHORUS_END_BEGINNER_YMCA = CHORUS_END_YMCA * SLOW_FACTOR

CHORUS_TIMING_BEGINNER_LYLLS = [(label, start * SLOW_FACTOR, end * SLOW_FACTOR) for (label, start, end) in CHORUS_TIMING_LYLLS]
CHORUS_END_BEGINNER_LYLLS = CHORUS_END_LYLLS * SLOW_FACTOR

SONG_CONFIGS = {
    "YMCA": {
        "labels": YMCA_LABELS,
        "audio_normal": AUDIO_PATH_YMCA_NORMAL,
        "audio_beginner": AUDIO_PATH_YMCA_SLOW,
        "timing_normal": CHORUS_TIMING_YMCA,
        "end_normal": CHORUS_END_YMCA,
        "timing_beginner": CHORUS_TIMING_BEGINNER_YMCA,
        "end_beginner": CHORUS_END_BEGINNER_YMCA,
    },
    "LYLLS": {
        "labels": LYLLS_LABELS,
        "audio_normal": AUDIO_PATH_LYLLS_NORMAL,
        "audio_beginner": AUDIO_PATH_LYLLS_SLOW,  
        "timing_normal": CHORUS_TIMING_LYLLS,
        "end_normal": CHORUS_END_LYLLS,
        # if you defined beginner timing:
        "timing_beginner": CHORUS_TIMING_BEGINNER_LYLLS,
        "end_beginner": CHORUS_END_BEGINNER_LYLLS,
    },
}


def extract_hand_vec(hand_landmarks):
    pts = np.array([(lm.x, lm.y) for lm in hand_landmarks.landmark], dtype=np.float32)
    origin = pts[0]
    pts = pts - origin
    scale = np.linalg.norm(pts[9]) + 1e-6
    pts = pts / scale
    return pts.flatten()

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


def get_expected_label(start_time, timing_list, chorus_end):
    """
    timing_list: list of (label, t_start, t_end) in seconds
    chorus_end: float, total duration
    """
    if start_time is None or not timing_list:
        return None, 0.0, 0.0

    elapsed = time.time() - start_time

    # If we overshoot the chorus, just return None
    if elapsed >= chorus_end:
        return None, 0.0, elapsed

    # Normal case: look for a window
    for (label, t0, t1) in timing_list:
        if t0 <= elapsed < t1:
            duration = max(1e-6, (t1 - t0))
            progress = (elapsed - t0) / duration
            return label, progress, elapsed

    # Fallbacks (try not to leave user with no expected label)
    first_label, first_start, first_end = timing_list[0]
    last_label, last_start, last_end = timing_list[-1]

    if elapsed < first_start:
        duration = max(1e-6, (first_end - first_start))
        progress = (elapsed - first_start) / duration
        progress = max(0.0, min(1.0, progress))
        return first_label, progress, elapsed

    duration = max(1e-6, (last_end - last_start))
    progress = (elapsed - last_start) / duration
    progress = max(0.0, min(1.0, progress))
    return last_label, progress, elapsed

def draw_top_bar(frame, expected, predicted, score, w):
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 70), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)
    cv2.putText(frame, f"Expected {expected}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255, 2))
    cv2.putText(frame, f"Predicted: {predicted} Score: {score:.2f}", (250, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return frame

def draw_bottom_bar(frame, h):
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h-40), (frame.shape[1], h), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)
    cv2.putText(frame, "Q - Quit, M - Menu, S - Start, N/B - Next/Back", (10, h -10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return frame

def start_performance(song_name, difficulty):
    cfg = SONG_CONFIGS[song_name]
    global current_wave, current_timing, current_chorus_end
    global per_letter_scores, start_time, playing, performance_level
    global templates, GHOST_COORDS, missing_for_current_song

    performance_level = difficulty

    # 1) Load templates for this song
    templates, GHOST_COORDS, missing_for_current_song = load_templates_for_song(song_name)
    if missing_for_current_song:
        print(f"[ERROR] Cannot start {song_name} – missing templates: {', '.join(missing_for_current_song)}")
        playing = False
        start_time = None
        current_timing = None
        current_chorus_end = 0.0
        return

    # 2) Choose audio + timing for this song & difficulty
    if difficulty == "beginner":
        audio_path = cfg.get("audio_beginner", cfg["audio_normal"])
        current_timing = cfg.get("timing_beginner", cfg["timing_normal"])
        current_chorus_end = cfg.get("end_beginner", cfg["end_normal"])
    else:
        audio_path = cfg["audio_normal"]
        current_timing = cfg["timing_normal"]
        current_chorus_end = cfg["end_normal"]

    # 3) Create WaveObject from path
    current_wave = sa.WaveObject.from_wave_file(str(audio_path))

    # 4) Reset scores
    per_letter_scores = {label: [] for label in cfg["labels"]}

    # 5) Start audio + timer
    play_obj = current_wave.play()
    start_time = time.time()
    playing = True

    print(f"[INFO] Performance started: {song_name} ({difficulty})")
    print(f"[DEBUG] timing windows={len(current_timing)}, end={current_chorus_end}")


def main():
    menu_stage = "welcome"
    good_streak = 0
    GOOD_STREAK_TARGET = 10

    mode = "menu"   # "menu", "tutorial", "performance", "summary"
    summary_lines = []
    tutorial_index = 0  # index into TUTORIAL_ORDER

    # Start with YMCA by default
    current_song = "YMCA"
    templates, GHOST_COORDS, missing = load_templates_for_song(current_song)

    if missing:
        print(f"[WARN] YMCA templates missing for: {', '.join(missing)}")
        print("      Run capture_templates.py in YMCA mode to record them.")

    # Per-song letter/word scores, reset when starting a performance
    per_letter_scores = {}

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open webcam.")
        return

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

            # -------------------- MENU SYSTEM --------------------
            if mode == "menu":

                frame[:] = 0  # black background for clean UI
                h, w = frame.shape[:2]

                # ========== STAGE 1: WELCOME ==========
                if menu_stage == "welcome":
                    cv2.putText(frame, "Welcome to", (w//4 - 150, h//2 - 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,0), 3)
                    cv2.putText(frame, "Sign-Along: Where Music Meets Sign Language", (w//4 - 170, h//2),
                                cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0,255,255), 4)

                    cv2.putText(frame, "Press any key to continue",
                                (w//2 - 230, h//2 + 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
                    cv2.imshow('Sign-Along', frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key != 255:
                        menu_stage = "song_select"
                    continue


                # ========== STAGE 2: SONG SELECT ==========
                if menu_stage == "song_select":
                    cv2.putText(frame, "Choose Your Song", (w//2 - 200, 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0,255,255), 3)

                    cv2.putText(frame, "1) YMCA (letters)", (80, 180),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
                    cv2.putText(frame, "2) Love You Like a Love Song (ASL signs)", (80, 240),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)

                    cv2.putText(frame, "Q Quit", (80, 340),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200,200,200), 2)

                    cv2.imshow("SignAlong", frame)
                    key = cv2.waitKey(1) & 0xFF

                    if key == ord("1"):
                        current_song = "YMCA"
                        templates, GHOST_COORDS, missing = load_templates_for_song(current_song)
                        if missing:
                            print(f"[WARN] YMCA templates missing for: {', '.join(missing)}")
                        else:
                            print("[INFO] YMCA templates loaded.")
                        menu_stage = "activity_select"

                    if key == ord("2"):
                        current_song = "LYLLS"
                        templates, GHOST_COORDS, missing = load_templates_for_song(current_song)
                        if missing:
                            print(f"[WARN] LYLLS templates missing for: {', '.join(missing)}")
                        else:
                            print("[INFO] LYLLS templates loaded.")
                        menu_stage = "activity_select"

                    elif key in [ord("q"), ord("Q")]:
                        break
                    continue


                # ========== STAGE 3: ACTIVITY SELECT ==========
                if menu_stage == "activity_select":
                    cv2.putText(frame, f"Selected Song: {current_song}", (60, 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0,255,255), 3)

                    cv2.putText(frame, "T Tutorial (learn signs)", (80, 180),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
                    cv2.putText(frame, "P Performance Mode", (80, 240),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)

                    cv2.putText(frame, "M Back to Song Select", (80, 320),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200,200,200), 2)

                    cv2.imshow("SignAlong", frame)
                    key = cv2.waitKey(1) & 0xFF

                    if key in [ord("t"), ord("T")]:
                        mode = "tutorial"
                        tutorial_index = 0
                        print("[INFO] Entering tutorial…")
                    elif key in [ord("p"), ord("P")]:
                        menu_stage = "difficulty_select"
                    elif key in [ord("m"), ord("M")]:
                        menu_stage = "song_select"
                    continue


                # ========== STAGE 4: DIFFICULTY SELECT ==========
                if menu_stage == "difficulty_select":
                    templates, GHOST_COORDS, missing = load_templates_for_song(current_song)
                    if missing:
                        print(f"[ERROR] Templates missing for {current_song}: {', '.join(missing)}")
                        print("Run capture_templates.py for this song, then try again.")
                        mode = "menu"
                        continue

                    cv2.putText(frame, f"Difficulty for {current_song}", (60, 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0,255,255), 3)

                    cv2.putText(frame, "1 Beginner (slow audio)", (80, 200),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
                    cv2.putText(frame, "2 Normal", (80, 260),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)

                    cv2.putText(frame, "M Back", (80, 340),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200,200,200), 2)

                    cv2.imshow("SignAlong", frame)
                    key = cv2.waitKey(1) & 0xFF

                    if key in [ord("q"), ord("Q")]:
                        break
                    if key in [ord("m"), ord("M")]:
                        mode = "menu"
                        print("[INFO] Back to menu from difficulty.")
                        continue

                    if key == ord("1"):
                        start_performance(current_song, "beginner")
                        mode = "performance"
                        continue

                    if key == ord("2"):
                        start_performance(current_song, "normal")
                        mode = "performance"
                        continue

                # END MENU SYSTEM

            # ---------- SUMMARY MODE ----------
            if mode == "summary":
                overlay = frame.copy()
                cv2.rectangle(overlay, (40, 40), (w - 40, h - 40), (0, 0, 0), -1)
                #frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
                frame = draw_bottom_bar(frame, h)
                y0 = 80
                for i, line in enumerate(summary_lines):
                    y_line = y0 + i * 30
                    cv2.putText(frame, line, (60, y_line),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

                cv2.putText(frame, "R Replay chorus", (60, h - 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(frame, "M Back to menu   Q - Quit", (60, h - 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                frame = draw_bottom_bar(frame, h)
                cv2.imshow(f"{current_song} ASL Performance Summary", frame)
                key = cv2.waitKey(1) & 0xFF

                if key in [ord("q"), ord("Q")]:
                    break
                if key in [ord("m"), ord("M")]:
                    mode = "menu"
                    print("[INFO] Back to menu.")
                if key in [ord("r"), ord("R")]:
                    start_performance(current_song, performance_level)
                    if playing:
                        mode = "performance"
                    print(f"[INFO] Chorus restarted from summary for {current_song} in {performance_level} mode.")
                    continue

            # ---------- TUTORIAL MODE ----------
            if mode == "tutorial":
                # Pick tutorial order based on current song
                if current_song == "YMCA":
                    TUTORIAL_ORDER = TUTORIAL_ORDER_YMCA
                else:
                    TUTORIAL_ORDER = TUTORIAL_ORDER_LYLLS

                result = hands.process(rgb)
                current_label = TUTORIAL_ORDER[tutorial_index]
                predicted_label = "-"
                score = 0.0

                # If this song's templates aren't ready, show message and bounce back
                _, _, missing = load_templates_for_song(current_song)
                if missing:
                    # simple console message
                    print(f"[ERROR] Templates missing for {current_song}: {', '.join(missing)}")
                    print("Run capture_templates.py for this song, then try again.")
                    
                    # on-screen message
                    overlay = frame.copy()
                    cv2.rectangle(overlay, (40, 40), (w - 40, h - 40), (0, 0, 0), -1)
                    frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
                    cv2.putText(frame, "Dataset not ready for this song.", (60, 120),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                    cv2.putText(frame, f"Missing: {', '.join(missing)}", (60, 170),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(frame, "Run capture_templates.py, then return.", (60, 220),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(frame, "Press M for menu.", (60, h - 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    cv2.putText(frame, f"Press Q to Quit", ((60, h - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2))

                    cv2.imshow("SignAlong", frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key in [ord('q'), ord('Q')]:
                        break
                    if key in [ord("m"), ord("M")]:
                        mode = "menu"
                    continue
                
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
                frame = draw_top_bar(frame, current_label, predicted_label, score, w)

                frame = draw_bottom_bar(frame, h)
                cv2.imshow(f"{current_song} ASL Tutor", frame)
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
                if key in [ord("p"), ord("P")]:
                    # go to difficulty selection screen
                    mode = "difficulty"
                    print("[INFO] Choose difficulty: 1 = Beginner (slow), 2 = Normal")
                    continue

                elif key in [ord("s"), ord("S")]:
                    # start performance immediately from tutorial, using current difficulty
                    # if missing_for_current_song:
                    #     print(f"[ERROR] Cannot start {current_song} – missing templates: {', '.join(missing_for_current_song)}")
                    #     mode = "menu"
                    #     continue
                    templates, GHOST_COORDS, missing = load_templates_for_song(current_song)

                    if missing:
                        print(f"[WARN] YMCA templates missing for: {', '.join(missing)}")
                        print("      Run capture_templates.py in YMCA mode to record them.")

                    start_performance(current_song, performance_level)
                    if playing:
                        mode = "performance"
                    print(f"[INFO] Performance mode started from tutorial for {current_song}, level={performance_level}.")
                    continue


            # ---------- PERFORMANCE MODE ----------
            if mode == "performance":
                # If performance wasn't initialized correctly, bail to menu
                if not playing or start_time is None or current_timing is None:
                    print("[ERROR] Performance state not initialized, returning to menu.")
                    mode = "menu"
                    continue

                result = hands.process(rgb)

                if playing and start_time is not None and current_timing is not None:
                    expected_label, progress, elapsed = get_expected_label(start_time, current_timing, current_chorus_end)
                else:
                    expected_label, progress, elapsed = None, 0.0, 0.0
                
                print(f"[DEBUG perf] playing={playing}, start_time={start_time}, timing_len={len(current_timing) if current_timing else 0}, elapsed={elapsed:.2f}, expected={expected_label}")

                predicted_label = "-"
                score = 0.0
                
                
                if result.multi_hand_landmarks: #and expected_label is not None:
                    hand_landmarks = result.multi_hand_landmarks[0]
                    # draw landmarks
                    for lm in hand_landmarks.landmark:
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        cv2.circle(frame, (cx, cy), 2, (0, 255, 0), -1)
                    vec = extract_hand_vec(hand_landmarks)
                    predicted_label, score, dist = classify(vec, templates)

                    if predicted_label == expected_label:
                        per_letter_scores[expected_label].append(score)
                ref_img = REF_IMAGES.get(expected_label)
                if ref_img is not None:
                    ref_h = 180
                    scale = ref_h/ ref_img.shape[0]
                    ref_w = int(ref_img.shape[1]*scale)
                    ref_thumb = cv2.resize(ref_img, (ref_w, ref_h))

                    y_off = 40
                    x_off = w - ref_w - 20
                    frame[y_off:y_off + ref_h, x_off:x_off + ref_w] = ref_thumb
                    ref_thumb = cv2.resize(ref_img, (ref_w, ref_h))
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
                frame = draw_top_bar(frame, expected_label, predicted_label, score, w)
                frame = draw_bottom_bar(frame, h)
                cv2.imshow(f"{current_song} ASL Tutor", frame)
                key = cv2.waitKey(1) & 0xFF

    
                if key in [ord("q"), ord("Q")]:
                    break

                # chorus finished -> go to summary
                if playing and elapsed >= current_chorus_end:
                    playing = False
                    mode = "summary"

                    summary_lines = []
                    summary_lines.append(f"{current_song} ASL Performance Summary")
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
