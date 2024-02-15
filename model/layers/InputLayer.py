from model.layers.Layer import *

class InputLayer(Layer):
    def __init__(self, inputs_size):
        self.X = None
        self.__H_DIM = inputs_size
    
    def set_input(self, inputs):
        self.X = np.asarray(inputs)
        
    def get_outs_number(self) -> int|tuple:
        return self.__H_DIM
    
    def get_results(self) -> np.ndarray:
        return self.X
