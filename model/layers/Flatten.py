from model.layers.Layer import *
from model.layers.InputLayer import InputLayer
    
from model.optimizers.Optimizer import Optimizer
from model.optimizers.Adam import Adam
from model.optimizers.SGD import SGD

from model.actiators.ActivatorsHolder import ActivatorsHolder


class Flatten(Layer):
    
    def __init__(self, in_:Layer = None, *args, **kwargs):
        self.outs = None
        self.__H_DIM = None
        if in_:
            
            self.prev_layer:Layer = in_
            self._a_y, self._a_x = self.prev_layer.get_outs_number()
            self.__H_DIM = self._a_x * self._a_y
            
    
    def setIn(self, in_):
        # print(type(in_))
        # print(in_.get_outs_number())
        self.prev_layer:Layer = in_
        self._a_y, self._a_x = self.prev_layer.get_outs_number()
        self.__H_DIM = self._a_x * self._a_y
        # print(self.__H_DIM)
    
    def recalculate(self):
        data = self.prev_layer.get_results()
        
        self.outs = np.asarray(list(map(lambda x: self.__to_vector(x), data)))
        # print('outs', self.outs)
    def get_results(self)  -> np.ndarray:
        return self.outs
    
    def get_outs_number(self) -> int|tuple:
        return self.__H_DIM
    
    def evaluate(self, y, is_last = True, learning_rate=0.00002, input_ = None, ce_type = 1):
        # print('y', y)
        matrix_data = np.asarray(list(map(lambda x: self.__to_matrix(x), y)))
        return self.prev_layer.evaluate(matrix_data, is_last=False, learning_rate=learning_rate)
    
    def __to_vector(self, matrix:np.array)->np.array:
        res = matrix.reshape(np.multiply(*matrix.shape),)
        # print(res)
        return res
    
    def __to_matrix(self, vector:np.array)->np.array:
        # print(vector)
        return vector.reshape(*self.prev_layer.get_outs_number())