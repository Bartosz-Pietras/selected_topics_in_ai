import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import seaborn as sns

gpus = tf.config.list_physical_devices('GPU')
tf.config.set_visible_devices(gpus[0], 'GPU')

X, Y = keras.datasets.mnist.load_data()

X_train, Y_train = X
X_test, Y_test = Y

example_image = X_train[0]
plt.imshow(example_image)
plt.show()

X_train = np.expand_dims(X_train, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)


Y_train = keras.utils.to_categorical(Y_train)
Y_test = keras.utils.to_categorical(Y_test)

#
# X_test = np.asarray(X_test).astype("float32").reshape((-1, 1))
# Y_test = np.asarray(Y_test).astype("float32").reshape((-1, 1))



model = keras.models.Sequential()
model.add(layers.Input(shape=X_train.shape[1:]))
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
model.add(layers.Dense(10, activation="softmax"))


model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=tf.keras.losses.CategoricalCrossentropy(),
    metrics=[tf.keras.metrics.Accuracy()],
)

model.fit(X_train, Y_train, batch_size=32, epochs=10, validation_data=(X_test, Y_test), verbose=2)
loss, metrics = model.evaluate(X_test, Y_test, batch_size=32, verbose=2)
print(loss)
print(metrics)
predictions = np.argmax(model.predict(X_test), axis=1)

print(Y_test.shape)
print(predictions.shape)

confusion_matrix = tf.math.confusion_matrix(labels=np.argmax(Y_test, axis=1), predictions=predictions)

print(confusion_matrix)

# sns.heatmap(confusion_matrix)
# plt.show()
