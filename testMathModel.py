from math import sin
from random import shuffle
from copy import copy
def calc(x1, x2):
    return 10*x1 + 100 / (x2**2 + 1)

x_input = [(i,j) for i in range(-10,10) for j in range(-10,10)]
ordered_X = copy(x_input)
ordered_Y = []
for x in ordered_X:
    ordered_Y.append(calc(x[0],x[1]))
    

shuffle(x_input)
y_input = []
for x in x_input:
    y_input.append(calc(x[0],x[1]))
    
from matplotlib import pyplot as plt

# plt.plot(y_input)

from model.Sequential import Sequential
from model.layers.Dense import Dense
from model.actiators.functional import *
from time import time
from model.tools.PlotTools import PlotTools

model = Sequential('adam', ALPHA=0.00015, type_='mean_squared_error')
model.add(Dense(72, linear, input_shape=2))
model.add(Dense(32, sin))
model.add(Dense(16, backLinear))
model.add(Dense(64, linear))
# model.add(Dense(32, inverseProportion))

model.add(Dense(1, linear))

data = []
for i in range(int(len(x_input)//3)*2):
    x = np.array([x_input[i]])
    y = np.array([y_input[i]])
    data.append([x, y])
print(data[0])
b=time()
model.fit(data, 10000, need_calculate_loss=False, batch_size=2)
print('fit time', time() - b)
results = [model.predict(np.array([ordered_X[i]]))[0][0] for i in range(len(ordered_X))]

print(model.predict(np.array([ordered_X[0]])), ordered_Y[0])

plt.plot(ordered_Y)
plt.plot(results)

plt.show()

PlotTools.plot_model(model, 'plot_model.png')
