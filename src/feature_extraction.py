from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
import numpy as np
import joblib
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix

IMG_SIZE = (128, 128)
BATCH = 32

train_gen = ImageDataGenerator(
    rescale=1./255, rotation_range=20, shear_range=0.4, zoom_range=0.2, horizontal_flip=True
).flow_from_directory(
    'data/train', target_size=IMG_SIZE, class_mode='sparse', batch_size=BATCH, shuffle=False)

val_test_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
    'data/val', target_size=IMG_SIZE, class_mode='sparse', batch_size=BATCH, shuffle=False)

test_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
    'data/test', target_size=IMG_SIZE, class_mode='sparse', batch_size=BATCH, shuffle=False)

base = VGG16(weights='imagenet', include_top=False, input_shape=(*IMG_SIZE, 3))
base.trainable = False
label_map = train_gen.class_indices
joblib.dump(label_map, "label_map.pkl")


feat_model = Model(inputs=base.input, outputs=base.get_layer('block5_pool').output)
feat_model.save("vgg16_feature_extractor.h5")
print("✅ Saved VGG16 feature extractor as vgg16_feature_extractor.h5")

def extract_features(generator):
    num_samples = generator.samples
    features = feat_model.predict(generator, steps=np.ceil(num_samples / BATCH), verbose=1)
    features = features.reshape(features.shape[0], -1)
    labels = generator.classes
    return features, labels

# 🧠 Extract features
X_train, y_train = extract_features(train_gen)
X_val, y_val = extract_features(val_test_gen)
X_test, y_test = extract_features(test_gen)

# 💾 Save features
##""np.save('X_train.npy', X_train)
#np.save('y_train.npy', y_train)
#np.save('X_val.npy', X_val)
#np.save('y_val.npy', y_val)
#np.save('X_test.npy', X_test)
#np.save('y_test.npy', y_test)




print("✅ Feature extraction and saving complete.")


