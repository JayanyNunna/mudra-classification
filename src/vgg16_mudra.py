import tensorflow as tf
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping

# Data directories
train_dir = '/home/jayanynunna/mudra-dataset/train'
val_dir = '/home/jayanynunna/mudra-dataset/val'
test_dir = '/home/jayanynunna/mudra-dataset/test'

img_size = (150, 150)
batch_size = 32

# Load datasets
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    labels='inferred',
    label_mode='int',
    image_size=img_size,
    batch_size=batch_size,
    shuffle=True
)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    val_dir,
    labels='inferred',
    label_mode='int',
    image_size=img_size,
    batch_size=batch_size,
    shuffle=False
)
test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    test_dir,
    labels='inferred',
    label_mode='int',
    image_size=img_size,
    batch_size=batch_size,
    shuffle=False
)

num_classes = len(train_ds.class_names)

# Convert datasets to numpy arrays
def dataset_to_numpy(dataset):
    images = []
    labels = []
    for batch in dataset:
        imgs, lbls = batch
        images.append(imgs.numpy())
        labels.append(lbls.numpy())
    images = np.concatenate(images)
    labels = np.concatenate(labels)
    return images, labels

train_images_np, train_labels_np = dataset_to_numpy(train_ds)
val_images_np, val_labels_np = dataset_to_numpy(val_ds)
test_images_np, test_labels_np = dataset_to_numpy(test_ds)

# One-hot encode labels
train_labels_cat = to_categorical(train_labels_np, num_classes)
val_labels_cat = to_categorical(val_labels_np, num_classes)
test_labels_cat = to_categorical(test_labels_np, num_classes)

# Preprocess images for VGG16
train_images_np = preprocess_input(train_images_np)
val_images_np = preprocess_input(val_images_np)
test_images_np = preprocess_input(test_images_np)

# Load VGG16 model
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(*img_size, 3))
base_model.trainable = False

# Build your model
model = models.Sequential([
    base_model,
    layers.Flatten(),
    layers.Dense(50, activation='relu'),
    layers.Dense(20, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])
# Create tensorboard callback (functionized because need to create a new one for each model)
import datetime
def create_tensorboard_callback(dir_name, experiment_name):
  log_dir = dir_name + "/" + experiment_name + "/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
  tensorboard_callback = tf.keras.callbacks.TensorBoard(
      log_dir=log_dir
  )
  print(f"Saving TensorBoard log files to: {log_dir}")
  return tensorboard_callback
# Compile and train
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
es = EarlyStopping(monitor='val_loss', patience=5)

model.fit(
    train_images_np, train_labels_cat,
    epochs=20,
    validation_data=(val_images_np, val_labels_cat),
    callbacks=[es]
)

# For final evaluation on test set
test_loss, test_acc = model.evaluate(test_images_np, test_labels_cat)
print(f'Test accuracy: {test_acc:.4f}')