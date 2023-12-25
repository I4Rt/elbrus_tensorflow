    
from model.optimizers.Optimizer import *

class Adam(Optimizer):
    counter = 0
    def __init__(self, b1 = 0.9, b2 = 0.999, e=10**-8):
        self.b1 = b1
        self.b2 = b2
        self.e = e
        
        self.m_prev = 0
        self.v_prev = 0
        self.t = 1
        
    def calc(self, input:np.array, learning_rate:float, dInput:np.array):
        """Осуществляет обновление воходных параметров input в соответствии c методом Aadam.
        
        Args:
            input: Входной набор изменяемых коэффцииентов
            learning_rate: коеффициент смещения 
            dInput: градиент, соответствующий входному набору весов
        
        Returns:
            Обновленое значение входных параметров
        """

        m_t = self.b1 * self.m_prev + (1 - self.b1)*dInput
        v_t = self.b2 * self.v_prev + (1 - self.b2)*(dInput**2)
        M_t = m_t / (1 - self.b1 ** self.t)
        V_t = v_t / (1 - self.b2 ** self.t)
        
        step = learning_rate * M_t / (np.sqrt(V_t) + self.e)
        
        res = input - step
        
        self.t += 1
        self.m_prev = m_t
        self.v_prev = v_t
        
        return res