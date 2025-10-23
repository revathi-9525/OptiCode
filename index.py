"""
Eye Gesture → Morse Code → Letter (Hold + Save + Speak Word Version)
File: eye_gesture_morse.py

Features:
- Eye gaze (Right = dot, Left = dash)
- Blink to decode Morse into a letter
- Automatically saves letters to 'morse_output.txt'
- Speaks the full decoded message aloud (not just one letter)

Run:
    pip install opencv-python mediapipe numpy pyttsx3
    python eye_gesture_morse.py
"""

import time
from collections import deque
import cv2
import mediapipe as mp
import numpy as np
import pyttsx3

# -------------------------
# Configuration
# -------------------------
CAMERA_ID = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# Timing and sensitivity settings
GAZE_STABILITY_FRAMES = 10
GAZE_HOLD_TIME = 1.0       # seconds to hold gaze before registering symbol
SYMBOL_DELAY = 1.0         # delay between symbols
SYMBOL_TIMEOUT = 6.0
BLINK_FRAMES_REQUIRED = 2

# Gaze thresholds
GAZE_RIGHT_THRESHOLD = 0.03
GAZE_LEFT_THRESHOLD = -0.03
BLINK_EAR_THRESHOLD = 0.015

# Output file
OUTPUT_FILE = "morse_output.txt"

# Morse dictionary
MORSE_CODE = {
    '.-': 'A', '-...': 'B', '-.-.': 'C', '-..': 'D', '.': 'E', '..-.': 'F',
    '--.': 'G', '....': 'H', '..': 'I', '.---': 'J', '-.-': 'K', '.-..': 'L',
    '--': 'M', '-.': 'N', '---': 'O', '.--.': 'P', '--.-': 'Q', '.-.': 'R',
    '...': 'S', '-': 'T', '..-': 'U', '...-': 'V', '.--': 'W', '-..-': 'X',
    '-.--': 'Y', '--..': 'Z',
    '-----': '0', '.----': '1', '..---': '2', '...--': '3', '....-': '4',
    '.....': '5', '-....': '6', '--...': '7', '---..': '8', '----.': '9'
}

# Mediapipe setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# Text-to-speech setup
engine = pyttsx3.init()
engine.setProperty('rate', 175)  # Speed (WPM)
engine.setProperty('volume', 1.0)

def speak_text(text):
    """Speaks the provided text aloud."""
    engine.say(text)
    engine.runAndWait()

# Landmark indices
LEFT_EYE_INNER = 133
LEFT_EYE_OUTER = 33
LEFT_UPPER_LID = 159
LEFT_LOWER_LID = 145
LEFT_IRIS = [468, 469, 470, 471]

RIGHT_EYE_INNER = 362
RIGHT_EYE_OUTER = 263
RIGHT_UPPER_LID = 386
RIGHT_LOWER_LID = 374
RIGHT_IRIS = [473, 474, 475, 476]

# Helper functions
def get_landmark_xy(landmarks, idx):
    lm = landmarks[idx]
    return np.array([lm.x, lm.y], dtype=np.float32)

def eye_opening(landmarks, upper_idx, lower_idx):
    up = get_landmark_xy(landmarks, upper_idx)
    low = get_landmark_xy(landmarks, lower_idx)
    return np.linalg.norm(up - low)

def iris_offset_x(landmarks, iris_idxs, eye_inner, eye_outer):
    iris_pts = np.array([get_landmark_xy(landmarks, i) for i in iris_idxs])
    iris_center = iris_pts.mean(axis=0)
    inner = get_landmark_xy(landmarks, eye_inner)
    outer = get_landmark_xy(landmarks, eye_outer)
    eye_center = (inner + outer) / 2.0
    eye_width = np.linalg.norm(inner - outer)
    if eye_width == 0:
        return 0.0
    return (iris_center[0] - eye_center[0]) / eye_width

def decode_morse(sequence):
    return MORSE_CODE.get(sequence, '?')

def save_to_file(text):
    """Append decoded text to output file."""
    with open(OUTPUT_FILE, "a") as f:
        f.write(text)
    print(f"[Saved] Added: {repr(text)}")

# -------------------------
# Main Loop
# -------------------------
cap = cv2.VideoCapture(CAMERA_ID)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

blink_queue = deque(maxlen=BLINK_FRAMES_REQUIRED)
current_sequence = ''
decoded_text = ''
last_symbol_time = None
blink_state = False
gaze_start_time = None
previous_gaze = 'CENTER'

print("Starting camera... Press 'q' to quit.")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera error.")
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)
        h, w = frame.shape[:2]

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0].landmark

            left_open = eye_opening(face_landmarks, LEFT_UPPER_LID, LEFT_LOWER_LID)
            right_open = eye_opening(face_landmarks, RIGHT_UPPER_LID, RIGHT_LOWER_LID)
            opening = (left_open + right_open) / 2.0

            left_offset = iris_offset_x(face_landmarks, LEFT_IRIS, LEFT_EYE_INNER, LEFT_EYE_OUTER)
            right_offset = iris_offset_x(face_landmarks, RIGHT_IRIS, RIGHT_EYE_INNER, RIGHT_EYE_OUTER)
            avg_offset = (left_offset + right_offset) / 2.0

            gaze_label = 'CENTER'
            if avg_offset > GAZE_RIGHT_THRESHOLD:
                gaze_label = 'RIGHT'
            elif avg_offset < GAZE_LEFT_THRESHOLD:
                gaze_label = 'LEFT'

            current_time = time.time()
            if gaze_label != previous_gaze:
                gaze_start_time = current_time
                previous_gaze = gaze_label

            gaze_duration = (current_time - gaze_start_time) if gaze_start_time else 0
            progress = min(gaze_duration / GAZE_HOLD_TIME, 1.0)

            # Register symbol after holding gaze
            if gaze_label in ['LEFT', 'RIGHT'] and gaze_duration >= GAZE_HOLD_TIME:
                symbol = '.' if gaze_label == 'RIGHT' else '-'
                if last_symbol_time is None or (current_time - last_symbol_time) > SYMBOL_DELAY:
                    current_sequence += symbol
                    last_symbol_time = current_time
                    print(f"Registered: {symbol} | Sequence: {current_sequence}")
                    gaze_start_time = None  # reset

            # Blink to decode sequence
            is_blinked = opening < BLINK_EAR_THRESHOLD
            blink_queue.append(is_blinked)
            if len(blink_queue) == blink_queue.maxlen and all(blink_queue):
                if not blink_state:
                    if current_sequence:
                        letter = decode_morse(current_sequence)
                        decoded_text += letter
                        save_to_file(letter)
                        print(f"Decoded '{current_sequence}' → '{letter}'")
                    else:
                        decoded_text += ' '
                        save_to_file(' ')
                        print("Blink = space")

                    # 🗣️ Speak entire decoded text so far
                    speak_text(decoded_text.strip())

                    current_sequence = ''
                    last_symbol_time = None
                blink_state = True
            else:
                blink_state = False

            # Reset sequence after timeout
            if last_symbol_time and (current_time - last_symbol_time) > SYMBOL_TIMEOUT and current_sequence:
                print(f"Timeout: cleared '{current_sequence}'")
                current_sequence = ''
                last_symbol_time = None

            # Draw UI overlays
            mp_drawing.draw_landmarks(
                frame, results.multi_face_landmarks[0], mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,0), thickness=1, circle_radius=1)
            )

            cv2.putText(frame, f"Gaze: {gaze_label}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
            cv2.putText(frame, f"Seq: {current_sequence}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
            cv2.putText(frame, f"Decoded: {decoded_text}", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200,200,0), 2)

            # Progress bar
            bar_length = int(200 * progress)
            cv2.rectangle(frame, (20, 140), (20 + bar_length, 160), (0, 255, 0), -1)
            cv2.rectangle(frame, (20, 140), (220, 160), (255, 255, 255), 2)
            cv2.putText(frame, "Hold gaze to register symbol", (20, 180),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
        else:
            cv2.putText(frame, "No face detected", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)

        cv2.imshow('Eye → Morse (Speak Word Version)', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
