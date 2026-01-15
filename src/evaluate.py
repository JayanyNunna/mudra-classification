import numpy as np
import joblib
from sklearn.metrics import classification_report, accuracy_score

# Load test data
X_test = np.load("X_test.npy")
y_test = np.load("y_test.npy")

# Load the trained model
svm = joblib.load("svm_mudra_model.pkl")

# Predict and evaluate
y_pred = svm.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("Test Accuracy:", acc)
print("Test Classification Report:\n", classification_report(y_test, y_pred))
