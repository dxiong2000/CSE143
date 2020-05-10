"""
Assignment 2
Jacob Baginski Doar, Edward Christenson, Daniel Xiong
simpleRNN.py
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
EPOCHS = 15
BATCH_SIZE = 64
EMBED_SIZE = 128
DROPOUT = 0.25
VOCAB_SIZE = 10000
NUM_OOV_BUCKETS = 1000
ACTIVATION = 'tanh'
OPTIMIZER = 'adam'

# Where to save our graphs
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "nlp"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)

# function for saving graphs
def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

# setting constant seeds
np.random.seed(42)
tf.random.set_seed(42)

# processing data
def process(dataset, vocab_size, num_oov_buckets):

    # helper function for string processing
    def preprocess(x_batch, y_batch):
        x_batch = tf.strings.substr(x_batch, 0, 300)
        x_batch = tf.strings.regex_replace(x_batch, rb"<br\s*/?>", b" ")
        x_batch = tf.strings.regex_replace(x_batch, b"[^a-zA-Z']", b" ")
        x_batch = tf.strings.split(x_batch)
        return x_batch.to_tensor(default_value=b"<pad>"), y_batch

    # encoding helper function
    def encode_words(x_batch, y_batch):
        return table.lookup(x_batch), y_batch

    # gets vocab
    vocabulary = Counter()
    for x_batch, y_batch in dataset.batch(BATCH_SIZE).map(preprocess):
        for review in x_batch:
            vocabulary.update(list(review.numpy()))

    # truncates vocab and encodes tokens into their integer ID's
    truncated_vocabulary = [word for word, count in vocabulary.most_common()[:vocab_size]]
    words = tf.constant(truncated_vocabulary)
    word_ids = tf.range(len(truncated_vocabulary), dtype=tf.int64)
    vocab_init = tf.lookup.KeyValueTensorInitializer(words, word_ids)
    table = tf.lookup.StaticVocabularyTable(vocab_init, num_oov_buckets)
    data_set = dataset.batch(BATCH_SIZE).map(preprocess)
    data_set = data_set.map(encode_words).prefetch(1)

    return data_set

# load data
train_data, validation_data, test_data = tfds.load(
    name="imdb_reviews", 
    split=('train[:60%]', 'train[60%:]', 'test'),
    as_supervised=True)

# process data sets
train_data = process(train_data, VOCAB_SIZE, NUM_OOV_BUCKETS)
validation_data = process(validation_data, VOCAB_SIZE, NUM_OOV_BUCKETS)
test_data = process(test_data, VOCAB_SIZE, NUM_OOV_BUCKETS)

# building model and training
model = keras.models.Sequential([
    keras.layers.Embedding(VOCAB_SIZE+NUM_OOV_BUCKETS, EMBED_SIZE),
    keras.layers.SimpleRNN(EMBED_SIZE, activation=ACTIVATION, use_bias=True,
        kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal',
        bias_initializer='zeros', dropout=DROPOUT, recurrent_dropout=0.0),
    keras.layers.Dense(1, activation='sigmoid')
])
model.compile(loss="binary_crossentropy", optimizer=OPTIMIZER, metrics=["accuracy"])
history = model.fit(train_data.shuffle(10000), validation_data=validation_data, epochs=EPOCHS)

final_accuracy = model.evaluate(test_data, steps=128)
print(final_accuracy)

# plot
plt.plot(history.history['accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.show()
