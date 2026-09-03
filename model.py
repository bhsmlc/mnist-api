import tensorflow as tf
from tensorflow.keras import models, layers
import matplotlib.pyplot as plt
import cv2
import numpy as np

data = tf.keras.datasets.mnist
dataset = data.load_data()
training = dataset[0]
testing = dataset[0]
x_train, y_train = training
x_test, y_test = testing


x_train_copy = x_train.copy()

for i in range(60000):
    img = x_train[i]

    resized_img = cv2.resize(img, (280, 280), interpolation=cv2.INTER_AREA)
    _, thresholded_img = cv2.threshold(
        resized_img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
    )

    x, y, w, h = cv2.boundingRect(thresholded_img)

    if w == 0 or h == 0:
        print("weird one")
        x_train[i] = (img / 255.0).reshape(28, 28)
        continue

    digit_crop = thresholded_img[y:y+h, x:x+w]

    if w > h:
        w = h
        digit_crop = cv2.resize(digit_crop, (w, h))

    scalar = 200 / h
    new_h = h * scalar
    new_w = w * scalar
    new_x = (280-new_w) // 2
    new_y = (280-new_h) // 2

    digit_crop = cv2.resize(
        digit_crop,
        (int(new_w), int(new_h)),
        interpolation=cv2.INTER_AREA
    )

    blank = np.full((280, 280), 0, dtype=np.uint8)

    blank[
        int(new_y):int(new_y)+int(new_h),
        int(new_x):int(new_x)+int(new_w)
    ] = digit_crop

    resized_img = cv2.resize(blank, (28, 28), interpolation=cv2.INTER_AREA)

    x_train[i] = (resized_img / 255.0).reshape(28, 28)


# x_train = 255 - x_train
# x_test = 255 - x_train

# x_train, x_test = x_train / 255.0, x_test / 255.0

x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

for i in range(10):
    plt.imshow(x_train[i], cmap="gray")
    plt.title(y_train[i])
    plt.show()
    plt.imshow(x_train_copy[i], cmap="gray")
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