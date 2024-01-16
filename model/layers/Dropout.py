from model.layers.Layer import *
from model.layers.InputLayer import InputLayer

from model.optimizers.Optimizer import Optimizer
from model.optimizers.Adam import Adam
from model.optimizers.SGD import SGD

from model.tools.MathTools import MathTools

from model.actiators.ActivatorsHolder import ActivatorsHolder

class Dropout(Layer):
    def __init__(self, percent, in_=None):
        self.indexes = np.array([])
        self.__percent = percent
        self.setIn(in_)
        
        
    def setIn(self, in_):
        if in_:
            self.prev_layer = in_
            self.target = int(self.prev_layer.get_outs_number()*(1-self.__percent))
            self.__H_DIM = self.prev_layer.get_outs_number()
        
            self.p = self.target / self.prev_layer.get_outs_number()
            self.q = 1-self.p
            
    def __selectNeurons(self, input_):
        
        self.indexes = np.random.choice(self.__H_DIM, self.target, replace=False)
        self.indexes.sort()
        # print('input size', len(input_), 'output size',self.__H_DIM)
        out = np.array( [input_[i]/self.q if i in self.indexes else np.zeros_like(input_[i]) for i in range(self.__H_DIM)])
        
        return out.T
        
    def __selectBackProp(self, dh):
        out = np.array( [dh[i]/self.q if i in self.indexes else np.zeros_like(dh[i]) for i in range(len(dh))])
        # print('out bp', len(out))
        return out
   
   
    def recalculate(self):
        self.outs = self.__selectNeurons(self.prev_layer.get_results().T)
        
        # print([len(i) for i in self.outs])
        
    def get_results(self)  -> np.ndarray:
        return self.outs
        
    def get_outs_number(self) -> int:
        return self.__H_DIM

    def evaluate(self, y, is_last = True, learning_rate=0.00002, input_ = None, ce_type = 1):
        dE_dh_prev = self.__selectBackProp(y) / self.q
        return self.prev_layer.evaluate(dE_dh_prev, is_last=False, learning_rate=learning_rate, ce_type=ce_type)
