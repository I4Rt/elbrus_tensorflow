from model.layers.Layer import *
from model.layers.InputLayer import InputLayer
    
from model.optimizers.Optimizer import Optimizer
from model.optimizers.Adam import Adam
from model.optimizers.SGD import SGD

from model.actiators.ActivatorsHolder import ActivatorsHolder


class Conv2D(Layer):
    def __init__(self, b_shape:tuple, activation, optimizer:Optimizer = SGD(), in_:Layer = None, input_shape:tuple|None = None):
        self.b:np.array = np.random.uniform(0., 1., b_shape)
        self._b_y, self._b_x = self.b.shape
        
        self.outs = None
        self.__H_DIM = None
        self.m, self.n = None, None
        
        if input_shape:
            if type(input_shape) == tuple:
                self.prev_layer = InputLayer(input_shape)
                self._a_y = input_shape[0]
                self._a_x = input_shape[1]
                self.m = self._a_y - self._b_y + 1
                self.n = self._a_x - self._b_x + 1
                self.__H_DIM = (self.m, self.n)
                self.outs = np.zeros((self.m,self.n))
            else:
                raise Exception('Wrong input shape param')
        elif in_:
            self.prev_layer:Layer = in_
            self._a_y, self._a_x = self.prev_layer.get_outs_number()
            self.m = self._a_y - self._b_y + 1
            self.n = self._a_x - self._b_x + 1
            self.__H_DIM = (self.m, self.n)
            
        
        if type(activation) == str:
            self.func = ActivatorsHolder.getFuncByName(activation)
        else:
            self.func = activation
        self.optimizer = optimizer
        
        
    def setIn(self, in_):
        self.prev_layer:Layer = in_
        self._a_y, self._a_x = self.prev_layer.get_outs_number()
        self.m = self._a_y - self._b_y + 1
        self.n = self._a_x - self._b_x + 1
        self.__H_DIM = (self.m, self.n)
        
        
            
    def recalculate(self):
        batch = self.prev_layer.get_results()
        # print(f'batch in conv2d: {type(batch)}, {batch}')
        res = []
        for a in batch:
            # print(f'a in conv2d: {type(a)}, {a}')
            data = np.zeros((self.m, self.n))
            for i in range(self.m):
                for j in range(self.n):
                    data[i][j] = np.sum(a[i:i+self._b_y, j:j+self._b_x] * self.b)
            res.append(data)
        self.t = np.asarray(res)
        self.outs= self.func(self.t)
        # print(self.outs)
        
    def get_results(self)  -> np.ndarray:
        return self.outs
    
    def get_outs_number(self) -> int|tuple:
        return self.__H_DIM
    
    def evaluate(self, y, is_last = True, learning_rate=0.00002, input_ = None, ce_type = 1):
        if is_last:
            raise Exception('Conv2D can not be last')
        prev_layer_data=self.prev_layer.get_results()
        
        dF_dh = y * self.func(self.t, dif=True)
        
        
        # dE_db = []
        for a in prev_layer_data:
            batch_dE_db = []
            for i in range(self._b_y):
                batch_dE_db.append([])
                for j in range(self._b_x):
                    # print(a[i:i+self.m,j:j+self.n])
                    batch_dE_db[i].append(np.sum(a[i:i+self.m,j:j+self.n] * dF_dh))
            batch_dE_db = np.asarray(batch_dE_db) 
            
            self.b = self.optimizer.calc(self.b, learning_rate, batch_dE_db) # TODO: do not work or not correct
            # print(self.b)
            
        dE_da = []
        for a in prev_layer_data:
            linked_b = []
            for j in range(self._a_y):
                linked_in_column = []
                linked_b_rows = self.b[:j+1, :][j-self._a_y:,:]
                # print('\nrow', j)
                for i in range(self._a_x):
                    linked_in_row = linked_b_rows[:, :i+1][:, i-self._a_x:]
                    # print(j,i, res_row)
                    linked_in_column.append(linked_in_row)
                linked_b.append(linked_in_column)

            linked_b = np.asarray(linked_b, dtype=object)

            # почленно перемножаем связанные B_x_y на A_x_y
            parsed = linked_b * a
            # высчитываем общую сумму для драдиента каждого числа A_x_y
            batch_dE_da = list(map(lambda row: list(map(lambda item: np.sum(item), row)), parsed))
            dE_da.append(batch_dE_da)
        dE_da = np.asarray(dE_da)
        # print('dE_da', dE_da)
        if type(self.prev_layer) == InputLayer:
            return None
        return self.prev_layer.evaluate(dE_da, is_last=False, learning_rate=learning_rate)
        