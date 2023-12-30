from model.layers.Layer import *
from model.optimizers.Optimizer import Optimizer
from model.optimizers.SGD import SGD

from model.tools.MathTools import MathTools


class BatchNormalization(Layer):
    
    def __init__(self, in_:Layer, neurons_count, activation_func, optimizer1:Optimizer = SGD(), optimizer2:Optimizer = SGD()):
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
        
        self.outs=MathTools.soft_results(h)
        
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