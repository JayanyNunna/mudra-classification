import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from collections import defaultdict

# Load data
X_train = np.load('X_train.npy')
y_train = np.load('y_train.npy')

# Target: 50 samples per class
samples_per_class = 50
class_indices = defaultdict(list)

# Group indices by class
for idx, label in enumerate(y_train):
    class_indices[int(label)].append(idx)

# Select samples_per_class indices from each class
selected_indices = []
for cls, indices in class_indices.items():
    if len(indices) >= samples_per_class:
        selected_indices.extend(indices[:samples_per_class])
    else:
        print(f"⚠️ Class {cls} has only {len(indices)} samples, using all.")
        selected_indices.extend(indices)

# Final subset
X_subset = X_train[selected_indices]
y_subset = y_train[selected_indices]

# Apply t-SNE
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_2D = tsne.fit_transform(X_subset)

# Plot
plt.figure(figsize=(12, 10))
scatter = plt.scatter(X_2D[:, 0], X_2D[:, 1], c=y_subset, cmap='nipy_spectral', s=10)
plt.title("t-SNE of VGG16 Features (Balanced 28-Class Sample)")
plt.colorbar(scatter, label="Class label")
plt.savefig("tsne_features_28_classes_balanced.png", dpi=300)
print("✅ Saved as tsne_features_28_classes_balanced.png")

