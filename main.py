import tensorflow as tf
from tensorflow.keras import models, layers
import matplotlib.pyplot as plt

data = tf.keras.datasets.mnist
dataset = data.load_data()
training = dataset[0]
testing = dataset[0]
x_train, y_train = training
x_test, y_test = testing

x_train = 255 - x_train
x_test = 255 - x_train

x_train, x_test = x_train / 255.0, x_test / 255.0

x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

x_train[x_train >= 0.5] = 1
x_train[x_train < 0.5] = 0

for i in range(10):
    plt.imshow(x_train[i], cmap="gray")
    plt.title(y_train[i])
    plt.show()

layer0 = layers.Input((28, 28, 1))
layer1 = layers.Conv2D(64, (3, 3), activation="relu")
layer2 = layers.MaxPooling2D((2, 2))
layer3 = layers.Conv2D(64, (3, 3), activation="relu")
layer4 = layers.MaxPooling2D((2, 2))
layer5 = layers.Flatten()
layer6 = layers.Dense(64, activation="relu")
layer7 = layers.Dense(10, activation="softmax")

model = models.Sequential()
model.add(layer0)
model.add(layer1)
model.add(layer2)
model.add(layer3)
model.add(layer4)
model.add(layer5)
model.add(layer6)
model.add(layer7)

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test), verbose=1)
model.evaluate(x_test, y_test, verbose=2)
model.save("mnistmodel.keras")

