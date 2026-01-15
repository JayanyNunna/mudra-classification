import sys
import os
import cv2
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
import mediapipe as mp
import matplotlib.pyplot as plt

# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# -----------------------------
# 🔧 CONFIGURATION
IMG_SIZE = (128, 128)
VGG_MODEL_PATH = "vgg16_feature_extractor.h5"
SVM_MODEL_PATH = "svm_mudra_model.pkl"
LABEL_MAP_PATH = "label_map.pkl"

# -----------------------------
# 🖐️ HAND CROPPING FUNCTION
def crop_hand(image):
    mp_hands = mp.solutions.hands
    with mp_hands.Hands(static_image_mode=True, max_num_hands=1) as hands:
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not results.multi_hand_landmarks:
            return None

        h, w, _ = image.shape
        hand = results.multi_hand_landmarks[0]
        x_coords = [lm.x for lm in hand.landmark]
        y_coords = [lm.y for lm in hand.landmark]

        xmin, xmax = int(min(x_coords) * w), int(max(x_coords) * w)
        ymin, ymax = int(min(y_coords) * h), int(max(y_coords) * h)

        # 👇 Scale padding relative to hand size
        width = xmax - xmin
        height = ymax - ymin
        pad_x = int(width * 0.4)
        pad_y = int(height * 0.4)

        # Apply padding and clamp to image bounds
        xmin = max(0, xmin - pad_x)
        xmax = min(w, xmax + pad_x)
        ymin = max(0, ymin - pad_y)
        ymax = min(h, ymax + pad_y)

        return image[ymin:ymax, xmin:xmax]


# -----------------------------
# 🚀 MAIN EXECUTION
def main(image_path):
    # 🔹 Step 1: Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Image not found at: {image_path}")
        sys.exit(1)

    # 🔹 Step 2: Crop the hand using MediaPipe
    cropped = crop_hand(image)
    if cropped is None:
        print("❌ No hand detected in the image.")

 # 🔹 Optional: Show and save cropped image using matplotlib
    import matplotlib.pyplot as plt
    plt.imshow(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
    plt.title("Cropped Image Sent to Model")
    plt.axis('off')
    plt.savefig("cropped_preview.png")
    plt.show()

    # 🔹 Step 3: Preview and save cropped image
    plt.imshow(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
    plt.title("Cropped Image Sent to Model")
    plt.axis('off')
    plt.savefig("cropped_preview.png")
    plt.show()

    # 🔹 Step 4: Preprocess the cropped image
    resized = cv2.resize(cropped, IMG_SIZE)
    normalized = resized / 255.0
    input_tensor = np.expand_dims(normalized, axis=0)  # Shape: (1, 128, 128, 3)

    # 🔹 Step 5: Load VGG16 feature extractor
    print("🔄 Loading VGG16 feature extractor...")
    vgg_model = load_model(VGG_MODEL_PATH)
    features = vgg_model.predict(input_tensor)
    features_flattened = features.reshape((1, -1))

    # 🔹 Step 6: Load SVM model and label map
    classifier = joblib.load(SVM_MODEL_PATH)
    label_map = joblib.load(LABEL_MAP_PATH)
    inverse_label_map = {v: k for k, v in label_map.items()}

    # 🔹 Step 7: Predict top 3 mudras using SVM
    probs = classifier.predict_proba(features_flattened)[0]
    top3_indices = np.argsort(probs)[-3:][::-1]

    print("🔝 Top 3 predictions:")
    for i in top3_indices:
        label = inverse_label_map.get(i, "Unknown")
        confidence = probs[i] * 100
        print(f"{label}: {confidence:.2f}%")

# -----------------------------
# 🧠 Run script from command line
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict_mudra.py <path_to_image>")
        sys.exit(1)

    image_path_arg = sys.argv[1]
    main(image_path_arg)

