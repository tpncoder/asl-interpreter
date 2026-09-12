"""
train.py - Train an ASL letter classifier using MediaPipe hand landmarks.

Dataset structure expected:
  asl_alphabet_train/asl_alphabet_train/<LETTER>/<LETTER><N>.jpg
  (e.g., asl_alphabet_train/asl_alphabet_train/A/A1.jpg ... A3000.jpg)

The model learns from 21 hand landmarks (x, y, z) extracted via MediaPipe,
NOT from raw pixels. This makes it lightweight and suitable for browser use.
"""

import os
import sys
import json
import numpy as np
import cv2
import mediapipe as mp
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ────────────────────────────────────────────
# 1. Configuration
# ─────────────────────────────────────────────
TRAIN_DIR = "asl_alphabet_train/asl_alphabet_train"
MODEL_PATH = "asl_model.pth"
LABEL_MAP_PATH = "label_map.json"
NUM_LANDMARKS = 21
COORDS_PER_LANDMARK = 3  # x, y, z
INPUT_DIM = NUM_LANDMARKS * COORDS_PER_LANDMARK  # 63

EPOCHS = 100
BATCH_SIZE = 64
LEARNING_RATE = 0.001
HIDDEN_DIM = 128

# ─────────────────────────────────────────────
# 2. MediaPipe Setup
# ─────────────────────────────────────────────
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ─────────────────────────────────────────────
# 3. Landmark Extraction
# ─────────────────────────────────────────────
def extract_landmarks(image_path):
    """
    Extract normalized hand landmarks from an image.
    Returns a flat array of 63 values (21 landmarks × 3 coords) or None.
    """
    img = cv2.imread(image_path)
    if img is None:
        return None

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks is None:
        return None

    # Take the first detected hand
    hand_landmarks = results.multi_hand_landmarks[0]
    landmarks = []
    for lm in hand_landmarks.landmark:
        landmarks.extend([lm.x, lm.y, lm.z])

    return np.array(landmarks, dtype=np.float32)


# ─────────────────────────────────────────────
# 4. Dataset Loading
# ─────────────────────────────────────────────
def load_dataset(train_dir):
    """
    Walk through folder structure and extract landmarks + labels.
    Folder names are used as class labels.
    """
    all_landmarks = []
    all_labels = []
    skipped = 0

    letters = sorted([
        d for d in os.listdir(train_dir)
        if os.path.isdir(os.path.join(train_dir, d))
    ])

    print(f"Found {len(letters)} classes: {letters}")

    for letter in letters:
        letter_dir = os.path.join(train_dir, letter)
        files = sorted(os.listdir(letter_dir))

        for fname in files:
            if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue

            fpath = os.path.join(letter_dir, fname)
            landmarks = extract_landmarks(fpath)

            if landmarks is not None:
                all_landmarks.append(landmarks)
                all_labels.append(letter)
            else:
                skipped += 1

            # Progress
            if len(all_landmarks) % 500 == 0:
                print(f"  Processed {len(all_landmarks)} samples "
                      f"(skipped {skipped} so far)...")

    print(f"\nTotal samples: {len(all_landmarks)}, Skipped: {skipped}")
    return np.array(all_landmarks), np.array(all_labels)


# ─────────────────────────────────────────────
# 5. Model Definition
# ─────────────────────────────────────────────
class ASLNet(nn.Module):
    """
    Simple MLP that takes 63 landmark coordinates and outputs class logits.
    """
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
# 6. Training Loop
# ─────────────────────────────────────────────
def train_model(X, y, label_encoder):
    num_classes = len(label_encoder.classes_)
    print(f"Number of classes: {num_classes}")

    # Encode labels
    y_encoded = label_encoder.transform(y)

    # Train/val split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    # Convert to tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.long)

    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Model, loss, optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = ASLNet(INPUT_DIM, HIDDEN_DIM, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # FIX: Removed 'verbose=True' for PyTorch 2.0+ compatibility
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )

    best_val_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        # ── Training ──
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * batch_X.size(0)
            _, predicted = outputs.max(1)
            train_total += batch_y.size(0)
            train_correct += predicted.eq(batch_y).sum().item()

        train_acc = train_correct / train_total
        train_loss_avg = train_loss / train_total

        # ── Validation ──
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_t.to(device))
            val_loss = criterion(val_outputs, y_val_t.to(device)).item()
            _, val_predicted = val_outputs.max(1)
            val_acc = val_predicted.eq(y_val_t.to(device)).sum().item() / y_val_t.size(0)

        scheduler.step(val_loss)

        # Get current learning rate for logging
        current_lr = optimizer.param_groups[0]['lr']

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch [{epoch:3d}/{EPOCHS}] LR: {current_lr:.6f} | "
                  f"Train Loss: {train_loss_avg:.4f} | Train Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"  → Saved best model (val_acc={val_acc:.4f})")

    print(f"\nTraining complete! Best validation accuracy: {best_val_acc:.4f}")
    return model


# ─────────────────────────────────────────────
# 7. Main
# ─────────────────────────────────────────────
def main():
    if not os.path.isdir(TRAIN_DIR):
        print(f"ERROR: Training directory not found: {TRAIN_DIR}")
        sys.exit(1)

    print("=" * 50)
    print("ASL Landmark Model Training")
    print("=" * 50)

    # Load data
    print("\n[1/4] Loading dataset and extracting landmarks...")
    X, y = load_dataset(TRAIN_DIR)

    if len(X) == 0:
        print("ERROR: No valid samples found. Check your dataset.")
        sys.exit(1)

    # Encode labels
    print("\n[2/4] Encoding labels...")
    label_encoder = LabelEncoder()
    label_encoder.fit(y)

    # Save label map for inference
    label_map = {int(i): str(c) for i, c in enumerate(label_encoder.classes_)}
    with open(LABEL_MAP_PATH, "w") as f:
        json.dump(label_map, f)
    print(f"Label map saved to {LABEL_MAP_PATH}")
    print(f"Classes: {list(label_encoder.classes_)}")

    # Train
    print("\n[3/4] Training model...")
    model = train_model(X, y, label_encoder)

    # Final evaluation
    print("\n[4/4] Final evaluation on validation set...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    y_encoded = label_encoder.transform(y)
    y_tensor = torch.tensor(y_encoded, dtype=torch.long).to(device)

    with torch.no_grad():
        outputs = model(X_tensor)
        _, predicted = outputs.max(1)
        acc = predicted.eq(y_tensor).sum().item() / y_tensor.size(0)

    print(f"Overall accuracy on full dataset: {acc:.4f}")
    print(f"\nModel saved to: {MODEL_PATH}")
    print("Done! Run test.py to use the model with your webcam.")


if __name__ == "__main__":
    main()