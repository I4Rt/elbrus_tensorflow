from model.layers.Layer import *
from model.layers.InputLayer import InputLayer

from model.optimizers.Optimizer import Optimizer
from model.optimizers.Adam import Adam
from model.optimizers.SGD import SGD

from model.tools.MathTools import MathTools

from model.actiators.ActivatorsHolder import ActivatorsHolder

import sys

class Dense(Layer):
    def __init__(self, units, activation, optimizer1:Optimizer = SGD(), optimizer2:Optimizer = SGD(), in_:Layer = None, input_shape = None):
        self.__H_DIM = units
        if input_shape:
            self.prev_layer = InputLayer(input_shape)
            self.W = np.random.rand(self.prev_layer.get_outs_number(), self.__H_DIM)
        elif in_:
            self.prev_layer = in_
            self.W = np.random.rand(self.prev_layer.get_outs_number(), self.__H_DIM)
        else:
            self.W = np.array([])
        self.b = np.random.rand(1, self.__H_DIM)
        self.W = (self.W - 0.5) * 2 * np.sqrt(1/self.__H_DIM)     # new
        self.b = (self.b - 0.5) * 2 * np.sqrt(1/self.__H_DIM)     # new
        
        if type(activation) == str:
            self.func = ActivatorsHolder.getFuncByName(activation)
        else:
            self.func = activation
        self.outs = None
        self.optimizer1 = optimizer1
        self.optimizer2 = optimizer2
        
    def setIn(self, in_):
        self.prev_layer = in_
        self.W = np.random.rand(self.prev_layer.get_outs_number(), self.__H_DIM)
   
    def recalculate(self):
        t = self.prev_layer.get_results() @ self.W + self.b
        h = self.func(t)
        self.t = t
        self.outs= MathTools.soft_results(h)
        # print('dence outs', self.outs)
        # print('prev shape', self.prev_layer.get_outs_number())
        
        
    def get_results(self)  -> np.ndarray:
        return self.outs
        
    def get_outs_number(self) -> int:
        return self.__H_DIM

    def evaluate(self, y, is_last = True, learning_rate=0.00002, input_ = None, ce_type = 1):
        # print('dense in', y)
        if is_last:
            # where is softmax???
            # y_full = to_full(y, self.__H_DIM)   # here Y params is a row matrix of wanted result class
            if ce_type in [1, 2] and self.func.__name__ in ['softmax', 'tanh', 'sigmoid', 'softZeroToOne']:
                if ce_type == 1:
                    if self.func.__name__ == 'softmax':
                        dE_dt = self.outs - y
                        # print('\n\n\nhere1\n\n\n')
                    # elif self.func.__name__ == 'sigmoid':
                    #     dE_dt = self.outs - y
                    # elif self.func.__name__ == 'tanh':
                    #     dE_dt = self.outs - y # 1 - y + self.outs - y/self.outs
                    else:
                        dE_dt = self.outs - y # TODO: wrong formula
                        # print('\n\n\nhere2\n\n\n')
                elif ce_type == 2:
                    if self.func.__name__ in ['sigmoid', 'softZeroToOne']:
                        dE_dt = -(y/(self.outs + 0.000001) + (y - 1)/(1 - self.outs + 0.000001))*self.func(self.t, dif = True)
                        # print('\n\n\nhere2\n\n\n')
                    elif self.func.__name__ in ['tanh', 'softmax']:
                        raise Exception('Not avaliable to use tanh, softmax in BCE')
                    else:
                        raise Exception('Not avaliable fuction for in BCE')
            elif ce_type in [3]:
                if ce_type == 3:
                    dE_dt = self.func(self.t, dif = True)*(self.outs - y)
                    # print('\n\n\nhere3\n\n\n')
            else:
                # print('mult res', y/self.outs)
                # print(y.shape, self.outs.shape)
                dE_dt = -y/self.outs * self.func(self.t, dif=True)
                # print('\n\n\nhere4\n\n\n')
            # print(y, self.outs, dE_dt, sep='\n')
            # dE_dt = np.sum(dE_dt, axis=0, keepdims=True)/len(dE_dt)
            # print('dE_dt',dE_dt)
            dE_dW = self.prev_layer.get_results().T @ dE_dt
            dE_db = np.sum(dE_dt, axis=0, keepdims=True)
            dE_dh_prev = dE_dt @ self.W.T
        else:
            dE_dt = y * self.func(self.t, dif=True) # here Y params is a derivative matrix of current layer outs
            # print('y', y, 'prevouts', self.prev_layer.get_results(), '\n de_dt', dE_dt)
            dE_dW = self.prev_layer.get_results().T @ dE_dt
            dE_db = np.sum(dE_dt, axis=0, keepdims=True)
            dE_dh_prev = dE_dt @ self.W.T
            # print('dE_dh_prev', dE_dh_prev)
        # print('prev', self.__class__, is_last, self.W.size, self.b.size)
        self.W = self.optimizer1.calc(self.W, learning_rate, dE_dW)
        self.b = self.optimizer2.calc(self.b, learning_rate, dE_db)
        # print('new', self.__class__, is_last, self.W.size, self.b.size)
        if type(self.prev_layer) == InputLayer:
            return None
        # print('dE_dh_prev', dE_dh_prev)
        # sys.exit()
        return self.prev_layer.evaluate(dE_dh_prev, is_last=False, learning_rate=learning_rate, ce_type=ce_type)
