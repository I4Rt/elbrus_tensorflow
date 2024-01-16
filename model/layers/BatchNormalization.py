from model.layers.Layer import *
from model.optimizers.Optimizer import Optimizer
from model.optimizers.SGD import SGD

from model.tools.MathTools import MathTools


class BatchNormalization(Layer):
    
    def __init__(self, in_:Layer=None, neurons_count=None, activation_func=None, optimizer1:Optimizer = SGD(), optimizer2:Optimizer = SGD()):
        self.prev_layer = None
        self.__H_DIM = None
        self.setIn(in_)
        
        
        self.func = self.__normalizer
        self.outs = None
        self.__epsila = 0.000001
        
        # ========== not ness ========== 
        # self.W = np.random.rand(self.prev_layer.get_outs_number(), self.__H_DIM)
        # self.b = np.random.rand(1, self.__H_DIM)
        # self.W = (self.W - 0.5) * 2 * np.sqrt(1/self.__H_DIM)     # new
        # self.b = (self.b - 0.5) * 2 * np.sqrt(1/self.__H_DIM)     # new
        
        # self.optimizer1 = optimizer1 
        # self.optimizer2 = optimizer2 
        # ========== not ness ========== 

    def setIn(self, in_):
        if in_:
            self.prev_layer = in_
            self.__H_DIM = self.prev_layer.get_outs_number()
    
    
    def __normalizer(self, ):

        o = self.prev_layer.get_results()
        batch_size = o.size / len(o)
        
        v=np.sum(o, axis = 0) # 
        
        self.mathematical_expectation = 1/batch_size * v #[m01, m02, ...]
        self.variance = 1/batch_size * np.sum(v - self.mathematical_expectation, axis=0)
        
        z = (v - self.mathematical_expectation)/np.sqrt(self.variance + self.__epsila)
        
        return z
   
    def recalculate(self):
        z = self.func()
        self.t = z
        h = z
        temp=MathTools.soft_results(h)
        
        self.outs = np.array([[item] for item in temp])
        
        
    def get_results(self)  -> np.ndarray:
        return self.outs
        
    def get_outs_number(self) -> int:
        return self.__H_DIM



    def evaluate(self, y, is_last = True, learning_rate=0.00002, input_ = None, ce_type = 1):
        dE_dh_prev = 1 / np.sqrt(self.variance + self.__epsila)
        return self.prev_layer.evaluate(dE_dh_prev, is_last=False)
    
    
    
