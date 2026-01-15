import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import time

# Load the saved features and labels
X_train = np.load("X_train.npy")
y_train = np.load("y_train.npy")
X_val = np.load("X_val.npy")
y_val = np.load("y_val.npy")

# Optional: print dataset shape
print(f"Train shape: {X_train.shape}, Validation shape: {X_val.shape}")

# Define a simple SVM classifier
svm = SVC(kernel='rbf', C=10, gamma='scale', probability=True)

# Measure time
start = time.time()
print("Training SVM...")
svm.fit(X_train, y_train)
print("Training complete in", round(time.time() - start, 2), "seconds")

# Evaluate on validation data
y_pred = svm.predict(X_val)
acc = accuracy_score(y_val, y_pred)
print("Validation Accuracy:", acc)
print("Classification Report:\n", classification_report(y_val, y_pred))

# Optional: Save the model if you want
import joblib
joblib.dump(svm, "svm_mudra_model.pkl")
print("SVM model saved as svm_mudra_model.pkl")
