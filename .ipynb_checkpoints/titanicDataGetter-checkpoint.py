import numpy as np
from ClassModules import *

from pandas import DataFrame

data = np.genfromtxt("train.csv", delimiter=",")
dataset = []

raw_pyhton_dataset = []


for row in data[1:]:
    raw_pyhton_dataset.append([row[2]/3, row[4], row[5] / 50, row[9] / 100, int(row[1])])


raw_dataset = DataFrame(raw_pyhton_dataset)
res = raw_dataset.dropna(axis=0)
res = res.to_numpy()

for i in range(714):
    row = res[i]
    data = [0., 0.]
    data[int(row[4])] = 1.0
    # data = 0.0
    # if int(row[4]):
    #     data = 1.0
    out = np.array(data)
    dataset_row = [np.array([[row[0], row[1], row[2], row[3]]]),  np.array(out)]
    
    dataset.append( dataset_row )
    # print([np.array([row[0], row[1], row[2], row[3]]), np.array([row[3]]) ] )
# print(raw_dataset)

# for i in res[:10]:
#     print(res[i])
    
print(len(dataset))

model = Model([[4], [20, relu], [10, relu], [2, softmax]], 'Adam', type_='crossentropy', ALPHA=0.001)
loss_arr, accuracy_arr = model.train(dataset, need_calculate_loss=True, need_calculate_accuracy=True, num_epochs=100)

for i in range(30):
    print(model.predict(dataset[i][0]), dataset[i][1])

print(model.calc_accuracy(dataset))


import matplotlib.pyplot as plt
plt.plot(loss_arr)
plt.plot(accuracy_arr)    
plt.show()



# data = np.genfromtxt("test.csv", delimiter=",")
# dataset = []

# raw_pyhton_dataset = []


# for row in data[1:]:
#     raw_pyhton_dataset.append([row[2]/3, row[4], row[5] / 50, row[9] / 100, int(row[1])])

# raw_dataset = DataFrame(raw_pyhton_dataset)
# res = raw_dataset.dropna(axis=0)
# res = res.to_numpy()

# for i in range(350):
#     row = res[i]
#     dataset.append( [np.array([row[0], row[1], row[2], row[3]]), np.array([row[3]])] )

# model.calc_accuracy(dataset)



# from keras.optimizers import Adam




# def Adam():
    
#     m.assign_add((gradient - m) * (1 - self.beta_1))    
#     v.assign_add((tf.square(gradient) - v) * (1 - self.beta_2))
#     if self.amsgrad:
#         v_hat = self._velocity_hats[self._index_dict[var_key]]
#         v_hat.assign(tf.maximum(v_hat, v))
#         v = v_hat
#     variable.assign_sub((m * alpha) / (tf.sqrt(v) + self.epsilon))