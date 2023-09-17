import tensorflow as tf
import keras
import numpy as np

model = keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
model.compile(optimizer='sgd', loss='mean_squared_error')

xs = np.array([-1,  0,  2, 3, 4, 10, 15], dtype=float)
ys = np.array([-1, -1,  3, 5, 7, 19, 29], dtype=float)

model.fit(xs, ys, epochs = 500)

print(model.predict([20]))

import math
import matplotlib.pyplot as plt

w = 3
b = 0

x = [-1,  0,  1, 2, 3, 4]
y = [-3, -1,  1, 3, 5, 7]

myY = []

for thisX in x:
    thisY = (w*thisX) + b
    myY.append(thisY)
    
print("real Y is " + str(y))
print("My Y is " + str(myY))

total_square_error = 0
size = len(y)
for i in range(0,size):
    square_error = (y[i] - myY[i]) **2
    total_square_error += square_error
    
total_square_error /= size

print("My loss is " + str(math.sqrt(total_square_error)))
