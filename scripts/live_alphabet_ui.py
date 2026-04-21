import cv2
import joblib
import mediapipe as mp
import numpy as np

MODEL_PATH = "models/alphabet/alphabet_mlp_model.pkl"
LABEL_ENCODER_PATH = "models/alphabet/alphabet_label_encoder.pkl"

# Load model and label encoder
model = joblib.load(MODEL_PATH)
label_encoder = joblib.load(LABEL_ENCODER_PATH)

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

def extract_landmarks_from_frame(frame, hands):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if not results.multi_hand_landmarks:
        return None, None

    hand_landmarks = results.multi_hand_landmarks[0]

    features = []
    for lm in hand_landmarks.landmark:
        features.append(lm.x)
        features.append(lm.y)
        features.append(lm.z)

    return np.array(features).reshape(1, -1), hand_landmarks

def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Could not open webcam.")
        return

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)

            features, hand_landmarks = extract_landmarks_from_frame(frame, hands)

            predicted_label = "No hand detected"

            if features is not None:
                pred_encoded = model.predict(features)[0]
                predicted_label = label_encoder.inverse_transform([pred_encoded])[0]

                # Draw landmarks
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

            # UI text
            cv2.rectangle(frame, (10, 10), (260, 70), (0, 0, 0), -1)
            cv2.putText(
                frame,
                f"Prediction: {predicted_label}",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            cv2.imshow("Alphabet Sign Test", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()