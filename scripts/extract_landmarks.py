import os
import cv2
import mediapipe as mp
import pandas as pd

# Paths
TRAIN_DIR = "data/alphabet/raw/train"
TEST_DIR = "data/alphabet/raw/test"
OUTPUT_TRAIN_CSV = "data/alphabet/landmarks/train_landmarks.csv"
OUTPUT_TEST_CSV = "data/alphabet/landmarks/test_landmarks.csv"

# MediaPipe setup
mp_hands = mp.solutions.hands

def extract_landmarks_from_image(image_path, hands):
    image = cv2.imread(image_path)
    if image is None:
        return None

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if not results.multi_hand_landmarks:
        return None

    hand_landmarks = results.multi_hand_landmarks[0]

    features = []
    for lm in hand_landmarks.landmark:
        features.append(lm.x)
        features.append(lm.y)
        features.append(lm.z)

    return features

def process_dataset(input_dir, output_csv):
    data = []

    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.5
    ) as hands:

        for label in sorted(os.listdir(input_dir)):
            label_path = os.path.join(input_dir, label)

            if not os.path.isdir(label_path):
                continue

            print(f"Processing label: {label}")

            for file_name in os.listdir(label_path):
                file_path = os.path.join(label_path, file_name)

                features = extract_landmarks_from_image(file_path, hands)

                if features is not None:
                    row = features + [label]
                    data.append(row)

    # Create columns
    columns = []
    for i in range(21):
        columns.extend([f"x{i}", f"y{i}", f"z{i}"])
    columns.append("label")

    df = pd.DataFrame(data, columns=columns)
    df.to_csv(output_csv, index=False)
    print(f"Saved: {output_csv}")
    print(f"Total samples: {len(df)}")

if __name__ == "__main__":
    os.makedirs("data/alphabet/landmarks", exist_ok=True)

    process_dataset(TRAIN_DIR, OUTPUT_TRAIN_CSV)
    process_dataset(TEST_DIR, OUTPUT_TEST_CSV)