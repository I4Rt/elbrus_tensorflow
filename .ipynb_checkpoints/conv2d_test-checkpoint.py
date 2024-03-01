from model.Sequential import *

from model.layers.Conv2D import *
from model.layers.Dense import *
from model.layers.Flatten import *

from model.actiators.functional import *

import numpy as np

data = []
for i in range(200):
    data.append([np.random.rand(24,24), np.asarray([i % 3])])
    
model = Sequential('adam', [Conv2D((5,5), relu, input_shape=(24,24)), Conv2D((5,5), relu), Flatten(), Dense(20, linear), Dense(1, relu)], ALPHA=0.01)
acc, loss, = model.fit(data, 20, batch_size=10, need_calculate_accuracy=True)


for i in range(10):
    print(model.predict(np.array([data[i][0]])), data[i][1])

import matplotlib.pyplot as plt

plt.plot(acc)    
plt.plot(loss)

plt.show()




# print(type(data), data[0])