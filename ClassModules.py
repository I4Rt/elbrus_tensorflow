from __future__ import annotations
import numpy as np
np.seterr(divide='ignore', invalid='ignore')


ts = []

class Optimizer:
    
    def calc(self, input):
        return input

class BaseOprimizer(Optimizer):
    def calc(self, input, learning_rate, dInput):
        res = input - learning_rate * dInput
        return res
    
class Adam(Optimizer):
    def __init__(self, b1 = 0.9, b2 = 0.999, e=10**-8):
        self.b1 = b1
        self.b2 = b2
        self.e = e
        
        self.m_prev = 0
        self.v_prev = 0
        self.t = 1
        
    def calc(self, input:np.array, learning_rate:float, dInput:np.array):
        # print(input)
        m_t = self.b1 * self.m_prev + (1 - self.b1)*dInput
        v_t = self.b2 * self.v_prev + (1 - self.b2)*(dInput**2)
        # print('m_t', m_t, sep='\n')
        # print('v_t', v_t, sep='\n')
        M_t = m_t / (1 - self.b1 ** self.t)
        V_t = v_t / (1 - self.b2 ** self.t)
        # print('M_t', M_t, sep='\n')
        # print('V_t', V_t, sep='\n')
        
        step = learning_rate * M_t / (np.sqrt(V_t) + self.e)
        

        res = input - step
        
        self.t += 1
        self.m_prev = m_t
        self.v_prev = v_t
        
        return res
    







def soft_results(data):
    # print(data)
    
    data[np.isnan(data)] = 0.00000001
    data[data == np.inf] = 999999
    data[data == -np.inf] = -99999
    # print('\n\n', data, '\n\n')
    # sleep(1)
    return data
    

def to_full(y, num_classes):
    """
    Returns array with 1 in class index position

    Parameters:
    -y: correct class index
    -num_classes: len of classes sequency
    """
    y_full = np.zeros((1, num_classes))
    y_full[0, y] = 1
    return y_full

class BatchNormalization:
    
    def __init__(self, in_:NeuronLayer|InputLayer, neurons_count, activation_func, optimizer1 = BaseOprimizer(), optimizer2 = BaseOprimizer()):
        self.prev_layer = in_
        
        self.__H_DIM = self.prev_layer.get_outs_number()
        self.W = np.random.rand(self.prev_layer.get_outs_number(), self.__H_DIM)
        self.b = np.random.rand(1, self.__H_DIM)
        self.W = (self.W - 0.5) * 2 * np.sqrt(1/self.__H_DIM)     # new
        self.b = (self.b - 0.5) * 2 * np.sqrt(1/self.__H_DIM)     # new
        
        self.func = self.__normalizer
        self.outs = None
        self.optimizer1 = optimizer1
        self.optimizer2 = optimizer2
        
    def __normalizer(self, ):
        o = self.prev_layer.get_results()
        self.mathematical_expectation = 1/o.size * np.sum(o)
        self.variance = 1/o.size * np.sum(o - self.mathematical_expectation)
        z = (o - self.mathematical_expectation)/np.sqrt(self.variance + 0.000001)
        return z
   
    def recalculate(self):
        
        z = self.func()
        self.t = z
        h = self.W * z + self.b
        
        self.outs=soft_results(h)
        
    def get_results(self)  -> np.ndarray:
        return self.outs
        
    def get_outs_number(self) -> int:
        return self.__H_DIM



    def evaluate(self, y, is_last = True, learning_rate=0.00002, input_ = None, ce_type = 1):
        # sleep(1)
        # out = z * W + b
        # z = func(h_prev)
        dE_dW = self.t
        dE_db = np.ones(self.get_outs_number())
        
        
        dE_dt = self.z
        
        d_variance = self.prev_layer.get_results() * 2 / self.prev_layer.get_outs_number() - 1 / self.get_outs_number**2
        d_mathematical_expectation = 1/self.outs
        v_e = self.variance + 0.000001
        
        dE_dh_prev = ((1-d_mathematical_expectation) * np.sqrt(v_e) - (dE_dt - self.mathematical_expectation)*d_variance / (2*np.sqrt(v_e)))/v_e
    
        
        self.W = self.optimizer1.calc(self.W, learning_rate, dE_dW)
        self.b = self.optimizer2.calc(self.b, learning_rate, dE_db)
        return self.prev_layer.evaluate(dE_dh_prev, is_last=False)
    
    
class NeuronLayer:
    
    def __init__(self, in_:NeuronLayer|InputLayer, neurons_count, activation_func, optimizer1 = BaseOprimizer(), optimizer2 = BaseOprimizer()):
        self.prev_layer = in_
        self.__H_DIM = neurons_count
        self.W = np.random.rand(self.prev_layer.get_outs_number(), self.__H_DIM)
        self.b = np.random.rand(1, self.__H_DIM)
        self.W = (self.W - 0.5) * 2 * np.sqrt(1/self.__H_DIM)     # new
        self.b = (self.b - 0.5) * 2 * np.sqrt(1/self.__H_DIM)     # new
        
        self.func = activation_func
        self.outs = None
        self.optimizer1 = optimizer1
        self.optimizer2 = optimizer2
        

   
    def recalculate(self):
        t = self.prev_layer.get_results() @ self.W + self.b
        h = self.func(t)
        self.t = t
        self.outs= soft_results(h)
        
    def get_results(self)  -> np.ndarray:
        return self.outs
        
    def get_outs_number(self) -> int:
        return self.__H_DIM



    def evaluate(self, y, is_last = True, learning_rate=0.00002, input_ = None, ce_type = 1):
        if is_last:       # where is softmax???
            # y_full = to_full(y, self.__H_DIM)   # here Y params is a row matrix of wanted result class
            if ce_type in [1, 2] and self.func.__name__ in ['softmax', 'tanh', 'sigmoid']:
                if ce_type == 1:
                    if self.func.__name__ == 'softmax':
                        dE_dt = self.outs - y
                    elif self.func.__name__ == 'sigmoid':
                        dE_dt = self.outs - y
                    elif self.func.__name__ == 'tanh':
                        dE_dt = self.outs - y # 1 - y + self.outs - y/self.outs
                elif ce_type == 2:
                    if self.func.__name__ == 'softmax':
                        dE_dt = self.outs - y
                    elif self.func.__name__ == 'sigmoid':
                        dE_dt =(1-self.outs)*y
                    elif self.func.__name__ == 'tanh':
                        dE_dt = (1/self.outs-self.outs)*y # 1 - y + self.outs - y/self.outs
            else:
                dE_dt = -y/self.outs * self.func(self.t, dif=True)
            dE_dW = self.prev_layer.get_results().T @ dE_dt
            dE_db = np.sum(dE_dt, axis=0, keepdims=True)
            dE_dh_prev = dE_dt @ self.W.T
        else:
            dE_dt = y * self.func(self.t, dif=True) # here Y params is a derivative matrix of current layer outs
            # print('y', y, 'prevouts', self.prev_layer.get_results(), '\n de_dt', dE_dt)
            dE_dW = self.prev_layer.get_results().T @ dE_dt
            dE_db = dE_dt
            dE_dh_prev = dE_dt @ self.W.T
            # print('dE_dh_prev', dE_dh_prev)
            
        self.W = self.optimizer1.calc(self.W, learning_rate, dE_dW)
        self.b = self.optimizer2.calc(self.b, learning_rate, dE_db)
        
        if type(self.prev_layer) == InputLayer:
            return None
        return self.prev_layer.evaluate(dE_dh_prev, is_last=False, learning_rate=learning_rate, ce_type=ce_type)
      
    
class InputLayer:
    
    def __init__(self, inputs_size):
        self.X = None
        self.__H_DIM = inputs_size
    
    def set_input(self, inputs):
        self.X = np.asarray(inputs)
        
    def get_outs_number(self) -> int:
        return self.__H_DIM
    
    def get_results(self) -> np.ndarray:
        return self.X

from time import sleep
class Model:
    
    def __init__(self, layers:list[list[int, function, ]], optimizer:Optimizer, type_='crossentropy', ALPHA = 0.0002 , class_number = 2):
        
        self.layers: list[InputLayer | NeuronLayer] = []
        try:
            if len(layers) < 2:
                raise IndexError()
            # [in_num, activator, *args]
            l0_data = layers.pop(0)
            l0 = InputLayer(l0_data[0])
            self.layers.append(l0)
            for l_data in layers:
                if optimizer == 'Adam':
                    local_optimizer1 = Adam()
                    local_optimizer2 = Adam()
                else:
                    local_optimizer1 = BaseOprimizer()
                    local_optimizer2 = BaseOprimizer()
                l = NeuronLayer(self.layers[-1], l_data[0], l_data[1], optimizer1=local_optimizer1, optimizer2= local_optimizer2)
                self.layers.append(l)
        except IndexError:
            raise Exception('Bad layer compose')
        # print(self.layers)
        self.type_ = type_
        self.learning_rate = ALPHA
        self.class_number = class_number
        
        
        
    def train(self, dataset, num_epochs, need_calculate_accuracy = False, need_calculate_loss = False):
        loss_arr = []
        accuracy_arr = []
        
        for ep in range(num_epochs):
            # print('ep', ep)
            
            # random.shuffle(dataset)
            for i in range(len(dataset)):
                x, y = dataset[i]
                if type(y) == np.ndarray:
                    # passed += 1
                    self.layers[0].set_input(x)
                    for layer in self.layers[1:]:
                        layer.recalculate()
                    
                    z=self.layers[-1].get_results()
                    

                    if self.type_ == 'crossentropy':                        
                        # Backward
                        self.layers[-1].evaluate(y, learning_rate=self.learning_rate, input_ = x, ce_type=1)
                    elif self.type_ == 'binary_crossentropy':
                        # Backward
                        self.layers[-1].evaluate(y, learning_rate=self.learning_rate, input_ = x, ce_type=2)
                    else:
                        raise Exception('Unknown model type')
                
            
            if need_calculate_loss:
                loss_arr.append(sparse_cross_entropy(z, y))
            if need_calculate_accuracy:
                accuracy_arr.append(self.calc_accuracy(dataset))
                
                    
        return loss_arr, accuracy_arr
                
    def predict(self, x):
        
        self.layers[0].set_input(x)
        for layer in self.layers[1:]:
            # print(layer.W)
            layer.recalculate()
        # print(self.layers[-1].t)
        z=self.layers[-1].get_results()
        return z
    
    def calc_accuracy(self, dataset):
        correct = 0
        passed = 0
        for x, y in dataset:
            if type(y) == np.ndarray:
                z = self.predict(x)
                if z.size > 1:
                    y_pred = np.argmax(z) # get position ?
                    if y_pred == np.argmax(y):
                        correct += 1
                else:
                    if np.max(z) >= 0.0 and np.max(y):
                        correct += 1
            else:
                passed += 1
        acc = correct / ( len(dataset) - passed )
        return acc

import random
import numpy as np
from sklearn import datasets
iris = datasets.load_iris()

INPUT_DIM = 4
OUT_DIM = 3
H_DIM = 10


def tanh(t, dif=False):
    if dif:
        return 1. - tanh(t) ** 2
    return (np.exp(t) - np.exp(-t)) / (np.exp(t) + np.exp(-t))

def relu(t, dif=False):
    if dif:
        return (t >= 0).astype(float)
    return np.maximum(t, 0)
def sigmoid(t, dif=False):
    if dif:
        return sigmoid(t)*(1-sigmoid(t))
    return 1/(1 + np.e **(-t))

def softmax(t, dif=False):
    # print(t)
    out = np.exp(t)
    if dif:
        return 1 / out
    # print(out)
    return out / np.sum(out, axis=1, keepdims=True)

def sparse_cross_entropy(z, y):
    try:
        return -np.log(z[0, y])
    except:
        return -np.log(z[0, 0])
        

def to_full(y, num_classes):
    y_full = np.zeros((1, num_classes))
    y_full[0, y] = 1
    return y_full


    



# def predict(x):
#     input_layer.set_input(x)
#     first_layer.recalculate()
#     second_layer.recalculate()
    
#     z=second_layer.get_results()
#     return z

# def calc_accuracy():
#     correct = 0
#     for x, y in dataset:
#         z = predict(x)
#         y_pred = np.argmax(z)
#         if y_pred == y:
#             correct += 1
#     acc = correct / len(dataset)
#     return acc

if __name__ == '__main__':
    
    # dataset = [(iris.data[i][None, ...], to_full(iris.target[i], 3)) for i in range(len(iris.target))]

    # input_layer = InputLayer(4)
    # first_layer = NeuronLayer(input_layer, 10, relu)
    # second_layer = NeuronLayer(first_layer, 3, softmax)
    

    # ALPHA = 0.0002
    # NUM_EPOCHS = 400
    # # BATCH_SIZE = 50

    # loss_arr = []
    # accuracy_arr = []
    # for ep in range(NUM_EPOCHS):
    #     random.shuffle(dataset)
    #     for i in range(len(dataset)):
    #         x, y = dataset[i]

    #         input_layer.set_input(x)
    #         first_layer.recalculate()
    #         second_layer.recalculate()
            
    #         z=second_layer.get_results()
    #         E = sparse_cross_entropy(z, y)

    #         # Backward
    #         second_layer.evaluate(y, learning_rate=ALPHA)


    #         # accuracy_arr.append(calc_accuracy())
    #         loss_arr.append(E)

    
    # model = Model([[4], [10, relu], [3, softmax]], class_number=3)
    
    # # print(model.predict(dataset[0][0]))
    
    # loss_arr, accuracy_arr = model.train(dataset, 400, need_calculate_loss=False, need_calculate_accuracy=False)
    # # accuracy = model.calc_accuracy(dataset)
    # # print("Accuracy:", accuracy)
    
    # print(model.predict(dataset[0][0]))
    
    # import matplotlib.pyplot as plt
    # plt.plot(loss_arr)
    # plt.plot(accuracy_arr)    
    # plt.show()
    
    # from random import randint
    # data = []
    # for i in range(1000):
    #     data.append([np.array([randint(0, 100) / 10 for j in range(10)]), np.array( -1 ** randint(0, 1))])
        
    from time import time
    # print(iris)
    dataset = [(iris.data[i][None, ...], to_full(iris.target[i], 3)) for i in range(len(iris.target))]
    random.shuffle(dataset)
    print(dataset[0])
    train = dataset[:len(dataset) - len(dataset)//15]
    test = dataset[-len(dataset)//15:]
    model2 = Model([[4], [20, relu], [3, sigmoid]], 'Adam', type_='crossentropy', ALPHA=0.0001,class_number=3)
    b = time()
    loss_arr, accuracy_arr = model2.train(train, 400, need_calculate_loss=False, need_calculate_accuracy=True)
    e = time()
    print('train time:', e - b, 'sec.')
    print('test accuraccy', model2.calc_accuracy(train), model2.calc_accuracy(test))
    
    import matplotlib.pyplot as plt
    plt.plot(loss_arr)
    plt.plot(accuracy_arr)    
    plt.show()
  
