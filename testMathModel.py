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



from ClassModules import *

model = Sequential('adam', ALPHA=0.00015, type_='mean_squared_error')
model.add(Dense(72, linear, input_shape=2))
model.add(Dense(32, cos))
model.add(Dense(16, backLinear))
model.add(Dense(32, linear))
# model.add(Dense(32, inverseProportion))

model.add(Dense(1, linear))

data = []
for i in range(int(len(x_input)//3)*2):
    x = np.array([x_input[i]])
    y = np.array([y_input[i]])
    data.append([x, y])


accuracy, loss = model.train(data, 100, need_calculate_loss=False)

results = [model.predict(np.array([ordered_X[i]]))[0][0] for i in range(len(ordered_X))]
plt.plot(ordered_Y)
plt.plot(results)


plt.show()



