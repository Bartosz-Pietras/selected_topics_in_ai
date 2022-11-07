import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt


X, Y = keras.datasets.mnist.load_data()

X_train, X_test = X
Y_train, Y_test = Y

example_image = X_train[0]
plt.imshow(example_image)
plt.show()


Y_train = keras.utils.to_categorical(Y_train)
Y_test = keras.utils.to_categorical(Y_test)


layer_list = [
    layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=X_train.shape),
    layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu'),
    layers.MaxPool2D(pool_size=(2, 2)),
    layers.MaxPool2D(pool_size=(2, 2)),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(rate=0.5),
    layers.Dense(9, activation='softmax'),
]

model = keras.Sequential(layer_list)

model.summary()

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=[tf.keras.metrics.BinaryAccuracy(),
                       tf.keras.metrics.FalseNegatives()])

model.fit()