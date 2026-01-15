import joblib
import numpy as np
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model, load_model

# Load your saved training data (if needed)
X_train = np.load("X_train.npy")
y_train = np.load("y_train.npy")

# 💾 Load trained SVM model if not already saved
# If you already saved svm previously, you can skip this
# Else load and re-train it just for saving
try:
    svm = joblib.load("svm_mudra_model.pkl")
    print("✅ SVM model loaded.")
except:
    from sklearn.svm import SVC
    svm = SVC(kernel='rbf', C=10, probability=True)
    svm.fit(X_train, y_train)
    joblib.dump(svm, "svm_mudra_model.pkl")
    print("✅ SVM model trained and saved as svm_mudra_model.pkl")

# 🧠 VGG16 Feature Extractor
base = VGG16(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
base.trainable = False
feat_model = Model(inputs=base.input, outputs=base.get_layer('block5_pool').output)
feat_model.save("vgg16_feature_extractor.h5")
print("✅ VGG16 feature extractor saved as vgg16_feature_extractor.h5")

# 🔖 Label map
# If you don't have access to train_gen anymore, you can recreate it:
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
    'data/train', target_size=(128, 128), class_mode='sparse', batch_size=32, shuffle=False)

label_map = train_gen.class_indices
joblib.dump(label_map, "label_map.pkl")
print("✅ Label map saved as label_map.pkl")


