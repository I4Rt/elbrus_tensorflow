import numpy as np
import random
from sklearn import datasets


from model.Sequential import Sequential
from model.layers.Dense import Dense
from model.actiators.functional import *
from model.tools.OneHotEncoderTools import OneHotEncoderTools



iris = datasets.load_iris()
dataset = [(iris.data[i][None, ...], OneHotEncoderTools.to_full(iris.target[i], 3)) for i in range(len(iris.target))]
random.shuffle(dataset)

train = dataset[:len(dataset) - len(dataset)//15]
test = dataset[-len(dataset)//15:]

#code
model2 = Sequential('adam', ALPHA=0.0001)
model2.add(Dense(20, relu, input_shape=4))
model2.add(Dense(10, relu))
model2.add(Dense(3, softmax))
loss_arr, accuracy_arr = model2.fit(train, 1000, need_calculate_loss=False, need_calculate_accuracy=True, batch_size=4)   
print('test accuraccy', model2.calc_accuracy(train), model2.calc_accuracy(test))



import matplotlib.pyplot as plt
plt.plot(loss_arr)
plt.plot(accuracy_arr)    
plt.show()