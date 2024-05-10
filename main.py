# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# %%
from tensorflow import keras

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# %%
x_train.shape

# %%
x_train[0]   # Each row and column has 28 pixels

# %%
plt.imshow(x_train[0])

# %%
# Normalizing the values to tune all the pixel values in between 0-1.
import tensorflow as tf

x_train = tf.keras.utils.normalize(x_train)
x_test = tf.keras.utils.normalize(x_test)

# %%
x_train[0]

# %%
from keras.models import Sequential # type: ignore
from keras.layers import Flatten, Dense # type: ignore

model = Sequential()

model.add(Flatten(input_shape = (28, 28)))
model.add(Dense(128, activation = 'relu'))
model.add(Dense(10, activation = 'softmax'))

# %%
model.compile(loss='sparse_categorical_crossentropy', optimizer= 'Adam', metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs = 25, validation_split= 0.2)

# %%
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])

# %%
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])

# %%
model.summary()

# %%
model.predict(x_test)  # probabilities of 0,1....9 resp.

# %%
y_prob = model.predict(x_test)
y_prob.argmax( axis= 1 )

# %%
y_pred = y_prob.argmax( axis= 1 )

# %%
from sklearn.metrics import accuracy_score
print("accuracy score: ", accuracy_score(y_test, y_pred))

# %%



