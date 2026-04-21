import os
import numpy as np
import tensorflow as tf
import gradio as gr
from PIL import Image

# ===== Paths =====
MODEL_PATH = "../models/alphabet/alphabet_cnn.keras"
TRAIN_DIR = "../data/alphabet/raw/train"

# ===== Settings =====
IMG_HEIGHT = 64
IMG_WIDTH = 64

# ===== Load model =====
model = tf.keras.models.load_model(MODEL_PATH)

# Get class names from train folder names
class_names = sorted(
    [folder for folder in os.listdir(TRAIN_DIR) if os.path.isdir(os.path.join(TRAIN_DIR, folder))]
)

def preprocess_image(image: Image.Image):
    image = image.convert("RGB")
    image = image.resize((IMG_WIDTH, IMG_HEIGHT))
    image_array = np.array(image).astype("float32") / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

def predict_sign(image):
    if image is None:
        return "No image provided.", {}

    image_array = preprocess_image(image)

    predictions = model.predict(image_array, verbose=0)[0]
    predicted_index = int(np.argmax(predictions))
    predicted_label = class_names[predicted_index]
    confidence = float(predictions[predicted_index])

    # Top 3 predictions
    top_3_indices = np.argsort(predictions)[-3:][::-1]
    top_3 = {
        class_names[i]: float(predictions[i])
        for i in top_3_indices
    }

    result_text = f"Prediction: {predicted_label}\nConfidence: {confidence:.2%}"
    return result_text, top_3

demo = gr.Interface(
    fn=predict_sign,
    inputs=gr.Image(type="pil", sources=["webcam"]),
    outputs=[
        gr.Textbox(label="Result"),
        gr.Label(label="Top 3 Predictions")
    ],
    title="Sign Language Alphabet Webcam Tester",
    description="Take a webcam snapshot and let the CNN predict the signed letter."
)

if __name__ == "__main__":
    demo.launch()