import numpy as np

from model.Sequential import Sequential
from model.layers.Dense import Dense
from model.layers.Dropout import Dropout
from model.actiators.functional import *
from model.tools.OneHotEncoderTools import OneHotEncoderTools
import random

import matplotlib.pyplot as plt

from model.Sequential import *
from model.layers.Conv2D import *
from model.layers.Flatten import *

from matplotlib import pyplot as plt

import threading
data = np.genfromtxt("datasets/train_mnist.csv", delimiter=",")
print('reed')




matrix_dataset = []
for row in data[1:]:
    matrix = np.array(row[1:]).reshape((28,28))/255
    encode_data = OneHotEncoderTools.to_full(int(row[0]), 10)
    matrix_dataset.append((matrix, encode_data))


random.shuffle(matrix_dataset)
cut_dataset = matrix_dataset[:3000]
           
print(cut_dataset[0])
sys.exit()


train_ = cut_dataset.copy()
model = Sequential('adam', [Conv2D((2,2), linear, input_shape=(28,28)), Conv2D((2,2), linear), Flatten(), Dense(100, relu), Dense(20, softZeroToOne), Dense(10, softmax)], ALPHA=0.0005)
acc, loss, = model.fit(train_, 30, batch_size=1, need_calculate_accuracy=True)
plt.plot(acc)
plt.plot(loss)
plt.show()

# print(model.predict(cut_dataset[0][0]), cut_dataset[0][1])
# print(model.calc_accuracy(cut_dataset))

