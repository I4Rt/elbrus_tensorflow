import numpy as np

class MathTools:
    
    @staticmethod
    def soft_results(data):
        data[np.isnan(data)] = 0.00000001
        data[data == np.inf] = 999999
        data[data == -np.inf] = -99999
        return data
    