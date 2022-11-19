import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import array_to_img, img_to_array
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import seaborn as sns



gpus = tf.config.list_physical_devices('GPU')
tf.config.set_visible_devices(gpus[0], 'GPU')

original_dataset = tf.keras.utils.image_dataset_from_directory(
    directory="cnn_data",
    labels="inferred",
    label_mode="categorical",
    color_mode="grayscale",
    image_size=(64, 64),
    shuffle=True,
)
augmented_dataset = tf.keras.utils.image_dataset_from_directory(
    directory="cnn_data_augmented",
    labels="inferred",
    label_mode="categorical",
    color_mode="grayscale",
    image_size=(64, 64),
    shuffle=True,
)


# X, Y = sign_dataset
# X_train, Y_train = X
# X_test, Y_test = Y
#
# example_image = X_train[0]
# plt.imshow(example_image)
# plt.show()
#
# X_train = np.expand_dims(X_train, axis=-1)
# X_test = np.expand_dims(X_test, axis=-1)
#
#
# Y_train = keras.utils.to_categorical(Y_train)
# Y_test = keras.utils.to_categorical(Y_test)
#
#
model = keras.models.Sequential()
model.add(layers.Input(shape=(64, 64, 1)))
model.add(
    layers.Conv2D(
        filters=32, kernel_size=(3, 3), activation="relu",
    )
)
model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu"))
model.add(layers.MaxPool2D(pool_size=(2, 2)))
model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), activation="relu"))
model.add(layers.MaxPool2D(pool_size=(2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation="relu"))
model.add(layers.Dropout(rate=0.5))
model.add(layers.Dense(7, activation="softmax"))


model.summary()
#
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=tf.keras.losses.CategoricalCrossentropy(),
    metrics=[tf.keras.metrics.Accuracy()],
    run_eagerly=True
)

original_model = model.fit(original_dataset, batch_size=1, epochs=100)
augmented_model = model.fit(augmented_dataset, batch_size=1, epochs=100)


# loss, metrics = model.evaluate(X_test, Y_test, batch_size=32, verbose=2)
# print(loss)
# print(metrics)
# predictions = np.argmax(model.predict(X_test), axis=1)
#
# print(Y_test.shape)
# print(predictions.shape)
#
# confusion_matrix = tf.math.confusion_matrix(labels=np.argmax(Y_test, axis=1), predictions=predictions)
#
# print(confusion_matrix)

# sns.heatmap(confusion_matrix)
# plt.show()
