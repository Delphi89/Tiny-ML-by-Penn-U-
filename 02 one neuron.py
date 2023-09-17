try:
    %tensorflow_version 2.x
except Exception:
    pass

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

INITIAL_W = 10.0
INITIAL_B = 10.0


class Model(object):
    def __init__(self):
        self.w = tf.Variable(INITIAL_W)
        self.b = tf.Variable(INITIAL_B)
        
    def __call__(self, x):
        return self.w * x + self.b
  
def loss(predicted_y, target_y):
    return tf.reduce_mean(tf.square(predicted_y - target_y))

def train(model, inputs, outputs, learning_rate):
    with tf.GradientTape() as t:
        current_loss = loss(model(inputs), outputs)
        dw, db = t.gradient(current_loss, [model.w, model.b])
        #print(dw,db)
        model.w.assign_sub(learning_rate * dw)
        model.b.assign_sub(learning_rate * db)
        #model.w = model.w - learning_rate * 0.1
        #model.b = model.b - learning_rate * 0.1
        return current_loss
                            
xs = [-1.0,  0.0,  1.0, 2.0, 3.0, 4.0]
ys = [-3.0, -1.0,  1.0, 3.0, 5.0, 7.0]    
LEARNING_RATE = 0.14

model = Model()

list_w, list_b = [], []
epochs = 50
losses = []

for epoch in range(epochs):
    list_w.append(model.w.numpy())
    list_b.append(model.b.numpy())
    current_loss = train(model, xs, ys, learning_rate=LEARNING_RATE)
    losses.append(current_loss)
    print('epoch %2d: w%1.2f b=%1.2f, loss=%2.5f' %
         (epoch, list_w[-1], list_b[-1], current_loss))
    
TRUE_w =  2.0
TRUE_b = -1.0
xaxis = range(epochs)
plt.plot(xaxis, list_w, 'r', xaxis, list_b, 'b')
plt.plot([TRUE_w] * epochs, 'r--', [TRUE_b] * epochs, 'b--')
plt.legend(['w','b','True w', 'True b'])
plt.show()
