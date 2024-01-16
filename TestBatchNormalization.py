from model.layers.Dense import Dense
from model.layers.BatchNormalization import BatchNormalization
from model.layers.InputLayer import InputLayer
from model.actiators.functional import linear
import numpy as np
from random import randint

in_ = np.array([[randint(0, 10) / 10 for i in range(10)] for i in range(10)])
# print(in_)

l1 = InputLayer(10)
l1.set_input(in_)

l_d = Dense(10, linear, in_ = l1)
l_b = BatchNormalization(l1)

l_d.recalculate()
l_b.recalculate()

print(l_d.get_results())
print(l_b.get_results())

l_b.evaluate(np.array([[randint(-10, 10) / 10] for i in range(10)]))