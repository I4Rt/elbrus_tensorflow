import numpy as np

class CrossEntropyTools:
    @staticmethod
    def sparse_cross_entropy(z, y):
        try:
            return -np.log(z[0, y])
        except:
            return -np.log(z[0, 0])
        