import sys
import cv2
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model

# Check for command line argument
if len(sys.argv) != 2:
    print("Usage: python predict_from_image.py <path_to_image>")
    sys.exit(1)

image_path = sys.argv[1]

# Load image
image = cv2.imread(image_path)
if image is None:
    print(f"❌ Image file not found: {image_path}")
    sys.exit(1)

# Resize to match model input size
image = cv2.resize(image, (128, 128))
image = image / 255.0  # Normalize
image = np.expand_dims(image, axis=0)  # Shape: (1, 128, 128, 3)

# Load VGG16 feature extractor
# Load VGG16 feature extractor
print("🔄 Loading model...")
vgg_model = load_model("vgg16_feature_extractor.h5")

# 🧠 DEBUG: Check model structure
vgg_model.summary()

# Continue with feature extraction
features = vgg_model.predict(image)


# Extract features
features = vgg_model.predict(image)
features_flattened = features.reshape((1, -1))  # Shape: (1, 8192)

#debug
#print("🔍 Feature vector shape:", features_flattened.shape)
#print("🧪 First few values:", features_flattened[0][:10])


# Load SVM classifier
classifier = joblib.load("svm_mudra_model.pkl")

# Load label map and invert it: index ➝ mudra name
label_map = joblib.load("label_map.pkl")
inverse_label_map = {v: k for k, v in label_map.items()}

#debug
import matplotlib.pyplot as plt
plt.imshow(image[0])
plt.title("Preprocessed Test Image")
plt.show()
plt.savefig("test2_image.png")


# Predict
predicted_label = classifier.predict(features_flattened)[0]
print(f" Raw predicted label: {predicted_label}")

# Get mudra name
mudra_name = inverse_label_map.get(int(predicted_label), "Unknown")

# Optional: remove "(1)" suffix if you want cleaner output
if mudra_name != "Unknown":
    mudra_name = mudra_name.replace("(1)", "").strip()



print(f" Predicted mudra: {mudra_name}")

# Predict class probabilities using decision_function (for SVM with probability=False)
if hasattr(classifier, "decision_function"):
    scores = classifier.decision_function(features_flattened)[0]  # Shape: (num_classes,)
elif hasattr(classifier, "predict_proba"):
    scores = classifier.predict_proba(features_flattened)[0]
else:
    print("Your SVM model does not support confidence scores.")
    sys.exit(1)

# Load the correct label map and reverse it
label_map = joblib.load("label_map.pkl")
inverse_label_map = {v: k for k, v in label_map.items()}

# Predict using probabilities (since you trained with probability=True)
probs = classifier.predict_proba(features_flattened)[0]
top3_indices = np.argsort(probs)[-3:][::-1]

print("🔝 Top 3 predictions:")
for i in top3_indices:
    label = inverse_label_map.get(i, "Unknown")
    confidence = probs[i] * 100
    print(f"{label}: {confidence:.2f}%")

