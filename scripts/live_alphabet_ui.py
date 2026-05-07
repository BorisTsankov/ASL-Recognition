import os
import math
import warnings
from collections import Counter, deque

import cv2
import joblib
import mediapipe as mp
import numpy as np
import pandas as pd

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="google.protobuf.symbol_database"
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "alphabet",
    "alphabet_best_landmark_model.pkl"
)

LABEL_ENCODER_PATH = os.path.join(
    BASE_DIR,
    "models",
    "alphabet",
    "alphabet_label_encoder.pkl"
)

FEATURE_COLUMNS = []

for i in range(21):
    FEATURE_COLUMNS.extend([f"x{i}", f"y{i}", f"z{i}"])

DISTANCE_PAIRS = [
    (4, 8),
    (8, 12),
    (12, 16),
    (16, 20),

    (0, 4),
    (0, 8),
    (0, 12),
    (0, 16),
    (0, 20),

    (5, 8),
    (9, 12),
    (13, 16),
    (17, 20),

    (4, 20),
    (8, 20),
]

for a, b in DISTANCE_PAIRS:
    FEATURE_COLUMNS.append(f"dist_{a}_{b}")

PREDICTION_HISTORY_SIZE = 10
CONFIDENCE_THRESHOLD = 0.50

model = joblib.load(MODEL_PATH)
label_encoder = joblib.load(LABEL_ENCODER_PATH)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


def normalize_landmarks(landmarks):
    wrist = landmarks[0]

    shifted = []

    for x, y, z in landmarks:
        shifted.append((x - wrist[0], y - wrist[1], z - wrist[2]))

    max_dist = 0.0

    for x, y, z in shifted:
        dist = math.sqrt(x ** 2 + y ** 2 + z ** 2)

        if dist > max_dist:
            max_dist = dist

    if max_dist == 0:
        return None

    normalized = []

    for x, y, z in shifted:
        normalized.extend([
            x / max_dist,
            y / max_dist,
            z / max_dist
        ])

    return normalized


def add_distance_features_to_single_sample(normalized_features):
    feature_dict = {}

    for i in range(21):
        base_index = i * 3

        feature_dict[f"x{i}"] = normalized_features[base_index]
        feature_dict[f"y{i}"] = normalized_features[base_index + 1]
        feature_dict[f"z{i}"] = normalized_features[base_index + 2]

    for a, b in DISTANCE_PAIRS:
        dx = feature_dict[f"x{a}"] - feature_dict[f"x{b}"]
        dy = feature_dict[f"y{a}"] - feature_dict[f"y{b}"]
        dz = feature_dict[f"z{a}"] - feature_dict[f"z{b}"]

        feature_dict[f"dist_{a}_{b}"] = math.sqrt(
            dx ** 2 + dy ** 2 + dz ** 2
        )

    feature_df = pd.DataFrame([feature_dict], columns=FEATURE_COLUMNS)

    return feature_df


def extract_landmarks_from_frame(frame, hands):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if not results.multi_hand_landmarks:
        return None, None

    hand_landmarks = results.multi_hand_landmarks[0]

    coords = []

    for lm in hand_landmarks.landmark:
        coords.append((lm.x, lm.y, lm.z))

    normalized_features = normalize_landmarks(coords)

    if normalized_features is None:
        return None, hand_landmarks

    feature_df = add_distance_features_to_single_sample(normalized_features)

    return feature_df, hand_landmarks


def get_smoothed_prediction(prediction_history):
    if not prediction_history:
        return "No hand detected"

    most_common = Counter(prediction_history).most_common(1)

    return most_common[0][0]


def get_prediction_label_and_confidence(features_df):
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features_df)[0]

        pred_encoded = int(np.argmax(probabilities))
        confidence = float(np.max(probabilities))

        raw_label = label_encoder.inverse_transform([pred_encoded])[0]

        return raw_label, confidence

    pred_encoded = int(model.predict(features_df)[0])
    raw_label = label_encoder.inverse_transform([pred_encoded])[0]

    return raw_label, 1.0


def draw_ui(frame, predicted_label, confidence):
    cv2.rectangle(
        frame,
        (10, 10),
        (390, 105),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        frame,
        f"Prediction: {predicted_label}",
        (20, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Confidence: {confidence:.2f}",
        (20, 78),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "Press Q to quit",
        (10, frame.shape[0] - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (200, 200, 200),
        1
    )


def main():
    print("Loaded model:", MODEL_PATH)
    print("Loaded label encoder:", LABEL_ENCODER_PATH)
    print("Feature count:", len(FEATURE_COLUMNS))
    print("Classes:", list(label_encoder.classes_))

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Could not open webcam.")
        return

    prediction_history = deque(maxlen=PREDICTION_HISTORY_SIZE)

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:

        while True:
            ret, frame = cap.read()

            if not ret:
                print("Failed to read frame from webcam.")
                break

            frame = cv2.flip(frame, 1)

            features_df, hand_landmarks = extract_landmarks_from_frame(
                frame,
                hands
            )

            predicted_label = "No hand detected"
            confidence = 0.0

            if features_df is not None:
                raw_label, confidence = get_prediction_label_and_confidence(
                    features_df
                )

                if confidence >= CONFIDENCE_THRESHOLD:
                    prediction_history.append(raw_label)
                    predicted_label = get_smoothed_prediction(
                        prediction_history
                    )
                else:
                    predicted_label = "Uncertain"

                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )
            else:
                prediction_history.clear()

            draw_ui(frame, predicted_label, confidence)

            cv2.imshow("Alphabet Sign Test", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()