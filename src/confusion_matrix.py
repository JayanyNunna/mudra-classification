import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

svm=joblib.load("svm_mudra_model.pkl")


# Load the true and predicted labels (if not already in memory)
y_val = np.load("y_val.npy")
X_val=np.load("X_val.npy")# True labels
y_pred = svm.predict(X_val)  # Optional: or use svm.predict(X_val) if not saved yet

# Generate the confusion matrix
cm = confusion_matrix(y_val, y_pred)

# Optional: Define class labels (0 to 27 for 28 classes)
class_names = [str(i) for i in range(28)]  # Replace with real class names if available

# Plot confusion matrix
plt.figure(figsize=(14, 12))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.title("Confusion Matrix (Validation Set)")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.tight_layout()
plt.show()
plt.savefig("confusion_matrix.png", dpi=300)
print("saved cm")