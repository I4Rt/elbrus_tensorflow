
from model.optimizers.Optimizer import *
    
class SGD(Optimizer):
    
    def calc(self, input, learning_rate, dInput):
        """Осуществляет обновление воходных параметров input в соответствии с методом стахастического градиентного спуска.
        
        Args:
            input: Входной набор изменяемых коэффцииентов
            learning_rate: коеффициент смещения 
            dInput: градиент, соответствующий входному набору весов
        
        Returns:
            Обновленое значение входных параметров
        """
        
        res = input - learning_rate * dInput
        return res
