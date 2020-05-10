"""
Assignment 2
Jacob Baginski Doar, Edward Christenson, Daniel Xiong
GRUPretrained.py
Due 2/16/2020
"""

import sys
assert sys.version_info >= (3, 5)
import os
import numpy as np
import sklearn
assert sklearn.__version__ >= "0.20"
import tensorflow as tf
from tensorflow import keras
import tensorflow_hub as hub
import tensorflow_datasets as tfds
assert tf.__version__ >= "2.0"
import matplotlib as mpl
import matplotlib.pyplot as plt
from collections import Counter

print("Version: ", tf.__version__)
print("Eager mode: ", tf.executing_eagerly())
print("Hub version: ", hub.__version__)
print("GPU is", "available" if tf.config.experimental.list_physical_devices("GPU") else "NOT AVAILABLE")


# Hyperparameters
EPOCHS = 10
BATCH_SIZE = 32
DROPOUT = 0.25
VALIDATION_STEPS = 250
STEPS_PER_EPOCH = 500

# setting constant seeds
np.random.seed(42)
tf.random.set_seed(42)

# load data
train_data, validation_data, test_data = tfds.load(
    name="imdb_reviews",
    split=('train[:60%]', 'train[60%:]', 'test'),
    as_supervised=True
)

# process data
def process(dataset):

    # helper function for string processing
    def preprocess(x_batch, y_batch):
        x_batch = tf.strings.substr(x_batch, 0, 300)
        return x_batch, y_batch

    # truncates vocab
    train_set = dataset.repeat().batch(BATCH_SIZE).map(preprocess)
    return train_set

train_data = process(train_data)
validation_data = process(validation_data)
test_data = process(test_data)

# building model and training
model = keras.Sequential([
    hub.KerasLayer("https://tfhub.dev/google/tf2-preview/nnlm-en-dim128/1", output_shape=[128], input_shape=[], dtype=tf.string, trainable=False),
    keras.layers.Reshape((1, 128)),
    keras.layers.GRU(64, return_sequences=True),
    keras.layers.GRU(64),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
history = model.fit(train_data, validation_data=validation_data, validation_steps=VALIDATION_STEPS,
                    epochs=EPOCHS, steps_per_epoch=STEPS_PER_EPOCH)

final_accuracy = model.evaluate(test_data, steps=128)
print(final_accuracy)


# plot
plt.plot(history.history['accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.show()
