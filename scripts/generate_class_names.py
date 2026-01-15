import pickle

# Load the label map
with open("label_map.pkl", "rb") as f:
    label_map = pickle.load(f)

# Reverse if it's {class_name: index}
if isinstance(next(iter(label_map.keys())), str):
    label_map = {v: k for k, v in label_map.items()}

# Sort by index and extract class names
sorted_classes = [label_map[i] for i in sorted(label_map)]

# Save to class_names.txt
with open("class_names.txt", "w") as f:
    for class_name in sorted_classes:
        f.write(str(class_name) + "\n")

print("✅ class_names.txt has been created successfully.")
