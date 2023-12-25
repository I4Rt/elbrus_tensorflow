from model.actiators.functional import *

class ActivatorsHolder:
    
    def getFuncByName(name:str):
        name = name.lower()
        if name == 'tanh':
            return tanh
        elif name =='relu':
            return relu
        elif name =='linear':
            return linear
        elif name == 'backlinear':
            return backLinear
        elif name =='inverseproportion':
            return inverseProportion
        elif name =='sin':
            return sin
        elif name =='cos':
            return cos
        elif name =='power3':
            return power3
        elif name =='sigmoid':
            return sigmoid
        elif name =='softmax':
            return softmax
        elif name =='softminusonetoplusone':
            return softMinusOneToPlusOne
        elif name =='softzerotoone':
            return softZeroToOne
        else:
            raise Exception('Wrong activator name')