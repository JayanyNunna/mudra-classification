import joblib
import pickle

# Path to the label map .pkl file
with open("label_map.pkl", "rb") as f:
    label_map = pickle.load(f)

print("🔖 Label map contents:")
for label, class_name in label_map.items():
    print(f"{label}: {class_name}")
