"""
test.py - Real-time ASL letter prediction using webcam + MediaPipe landmarks.

Loads the trained model from train.py and classifies hand gestures in real-time.
Draws landmarks on the frame and displays the predicted letter.

Requirements:
  - Model file: asl_model.pth  (from train.py)
  - Label map:  label_map.json (from train.py)
"""

import os
import sys
import json
import cv2
import numpy as np
import mediapipe as mp
import torch
import torch.nn as nn

# ─────────────────────────────────────────────
# 1. Configuration
# ─────────────────────────────────────────────
MODEL_PATH = "./asl_model.pth"
LABEL_MAP_PATH = "label_map.json"
NUM_LANDMARKS = 21
COORDS_PER_LANDMARK = 3
INPUT_DIM = NUM_LANDMARKS * COORDS_PER_LANDMARK  # 63
HIDDEN_DIM = 128

WEBCAM_INDEX = 0
CONFIDENCE_THRESHOLD = 0.6  # Min confidence to display prediction

# ─────────────────────────────────────────────
# 2. Model Definition (must match train.py)
# ─────────────────────────────────────────────
class ASLNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(ASLNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.network(x)


# ─────────────────────────────────────────────
# 3. Landmark Extraction (live video)
# ─────────────────────────────────────────────
def get_landmarks_from_frame(frame, hands):
    """
    Extract normalized landmarks from a webcam frame.
    Returns flat array of 63 values or None if no hand detected.
    """
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks is None:
        return None, None

    hand_landmarks = results.multi_hand_landmarks[0]
    landmarks = []
    for lm in hand_landmarks.landmark:
        landmarks.extend([lm.x, lm.y, lm.z])

    return np.array(landmarks, dtype=np.float32), hand_landmarks


# ─────────────────────────────────────────────
# 4. Drawing Utilities
# ─────────────────────────────────────────────
def draw_landmarks(frame, hand_landmarks, mp_drawing, mp_hands):
    """Draw hand landmarks and connections on the frame."""
    mp_drawing.draw_landmarks(
        frame,
        hand_landmarks,
        mp_hands.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
        mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=1)
    )


def draw_prediction(frame, letter, confidence, position=(10, 40)):
    """Draw the predicted letter and confidence on the frame."""
    text = f"Prediction: {letter}"
    conf_text = f"Confidence: {confidence:.1%}"

    # Background box
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)
    cv2.rectangle(frame, (position[0] - 5, position[1] - th - 5),
                  (position[0] + tw + 10, position[1] + 10),
                  (0, 0, 0), -1)

    cv2.putText(frame, text, position,
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, conf_text, (position[0], position[1] + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1, cv2.LINE_AA)


# ─────────────────────────────────────────────
# 5. Main
# ─────────────────────────────────────────────
def main():
    # ── Load model and label map ──
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model file not found: {MODEL_PATH}")
        print("Run train.py first.")
        sys.exit(1)

    if not os.path.exists(LABEL_MAP_PATH):
        print(f"ERROR: Label map not found: {LABEL_MAP_PATH}")
        sys.exit(1)

    with open(LABEL_MAP_PATH, "r") as f:
        label_map = json.load(f)

    num_classes = len(label_map)
    print(f"Loaded {num_classes} classes: {list(label_map.values())}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ASLNet(INPUT_DIM, HIDDEN_DIM, num_classes).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print(f"Model loaded on {device}")

    # ── MediaPipe setup ──
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )

    # ── Webcam ──
    cap = cv2.VideoCapture(WEBCAM_INDEX)
    if not cap.isOpened():
        print(f"ERROR: Could not open webcam {WEBCAM_INDEX}")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("\n" + "=" * 50)
    print("ASL Recognition - Live Demo")
    print("=" * 50)
    print("Controls:")
    print("  [Q] or [Esc] - Quit")
    print("  [S]          - Save snapshot")
    print("=" * 50 + "\n")

    # Smoothing buffer for stable predictions
    prediction_buffer = []
    BUFFER_SIZE = 5

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        frame_count += 1

        # Flip horizontally for mirror effect
        frame = cv2.flip(frame, 1)

        # Extract landmarks
        landmarks, hand_landmarks = get_landmarks_from_frame(frame, hands)

        if landmarks is not None:
            # Draw landmarks
            draw_landmarks(frame, hand_landmarks, mp_drawing, mp_hands)

            # Predict
            landmark_tensor = torch.tensor(
                landmarks, dtype=torch.float32
            ).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(landmark_tensor)
                probs = torch.softmax(output, dim=1)
                confidence, predicted_idx = probs.max(1)

                letter = label_map[str(predicted_idx.item())]
                conf_val = confidence.item()

            # Smooth predictions with buffer
            prediction_buffer.append((letter, conf_val))
            if len(prediction_buffer) > BUFFER_SIZE:
                prediction_buffer.pop(0)

            # Only show if confident enough
            if conf_val >= CONFIDENCE_THRESHOLD:
                # Get most common prediction in buffer
                from collections import Counter
                letters_only = [p[0] for p in prediction_buffer]
                most_common_letter = Counter(letters_only).most_common(1)[0][0]
                avg_conf = np.mean([p[1] for p in prediction_buffer if p[0] == most_common_letter])

                draw_prediction(frame, most_common_letter, avg_conf)
        else:
            # No hand detected
            cv2.putText(frame, "No hand detected", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            prediction_buffer.clear()

        # Instructions overlay
        cv2.putText(frame, "[Q]uit  [S]napshot", (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1, cv2.LINE_AA)

        # Show frame
        cv2.imshow("ASL Recognition", frame)

        # Key handling
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q'), 27):  # q or Esc
            break
        elif key in (ord('s'), ord('S')):
            snapshot_name = f"snapshot_{frame_count:05d}.jpg"
            cv2.imwrite(snapshot_name, frame)
            print(f"Snapshot saved: {snapshot_name}")

    # Cleanup
    hands.close()
    cap.release()
    cv2.destroyAllWindows()
    print("\nSession ended.")


if __name__ == "__main__":
    main()