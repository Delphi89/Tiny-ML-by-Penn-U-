import tensorflow as tf
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt

data = tf.keras.datasets.mnist
(training_images, training_labels), (val_images, val_labels) = data.load_data()

print("shape of training images: ", training_images.shape)
print("shape of training labels: ", training_labels.shape)
print("shape of validation images: ", val_images.shape)
print("shape of validation labels: ", val_labels.shape)

print(training_images[0])
print()
print("class: ", training_labels[0])

plt.imshow(training_images[0], cmap='gray')
print("Class: ", training_labels[0])
print()

plt.imshow(training_images[1], cmap='gray')
print("Class: ", training_labels[1])
print()


training_images = training_images / 255.0
val_images = val_images / 255.0

print(training_images[0])

model = tf.keras.models.Sequential([tf.keras.layers.Flatten(input_shape=(28,28)),
                                    tf.keras.layers.Dense(20,activation=tf.nn.relu),
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])

OPT = 'adam'
LOSS = 'sparse_categorical_crossentropy'

model.compile(optimizer = OPT, loss = LOSS, metrics = ['accuracy'])

model.fit(training_images, training_labels, epochs = 20, validation_data = (val_images, val_labels))

model.evaluate(val_images, val_labels)


classifications = model.predict(val_images)
print(clasifications[0])
print("predicted class: ", np.argmax(clasifications[0]))
print("The actual class: ", val_labels[0])


plt.imshow(val_images[0], cmap = 'gray')
