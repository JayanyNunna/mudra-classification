import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.utils import to_categorical

(train_ds,train_labels), (test_ds,test_labels) = tfds.load("tf_flowers",split=["train[:70%]", "train[70%:]"],batch_size=-1,as_supervised=True)

#resizing images
train_ds=tf.image.resize(train_ds,(150,150))
test_ds=tf.image.resize(test_ds,(150,150))

import numpy as np

# Convert to numpy array first
train_labels_np = np.array(train_labels)

# Get number of classes from unique labels
num_classes = len(np.unique(train_labels_np))

#transforming lables to correct format
train_labels = to_categorical(train_labels_np, num_classes)
test_labels=to_categorical(test_labels, num_classes)


#loading vgg16 model

from tensorflow.keras.applications.vgg16 import  VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input

#loading model
#include top is false cause the top layer has 1000 classes, top layer ypu build for your own use case

base_model=VGG16(weights='imagenet',include_top=False,input_shape=train_ds[0].shape)
base_model.trainable=False #you arent training the convolusonal layers

#preprocessing input
train_ds=preprocess_input(train_ds)
test_ds=preprocess_input(test_ds)
#adding last layers for our specific problem
from tensorflow.keras import layers,models



flatten_layer = layers.Flatten()
dense_layer_1=layers.Dense(50,activation='relu')#50 neurons relu activation
dense_layer_2=layers.Dense(20,activation='relu')#20 neurons relu activation
prediction_layer=layers.Dense(5,activation='softmax')#5 neuroms softmax activation

model=models.Sequential([
    base_model,
    flatten_layer,
    dense_layer_1,
    dense_layer_2,
    prediction_layer
])
#first put in base model that is vgg that we loaded then the flattened layer and the dense layers and the prediction layer


#complile and fit the model
from tensorflow.keras.callbacks import EarlyStopping

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
es = EarlyStopping(monitor='val_loss', patience=5)
model.fit(train_ds, train_labels, epochs=20, validation_data=(test_ds, test_labels), callbacks=[es])

