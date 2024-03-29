from __future__ import annotations

import json

import h5py
import numpy as np
from time import sleep, time
import sys

np.seterr(divide='ignore', invalid='ignore')

ts = []

class Optimizer:
    
    def __init__(self, *args, **kwargs) -> None:
        pass
    
    def calc(self, input):
        return input

class BaseOprimizer(Optimizer):
    def calc(self, input, learning_rate, dInput):
        res = input - learning_rate * dInput
        return res
    
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
        # Adam.counter += 1
        # if Adam.counter > 6:
        #     sys.exit()
        m_t = self.b1 * self.m_prev + (1 - self.b1)*dInput
        v_t = self.b2 * self.v_prev + (1 - self.b2)*(dInput**2)
        # print('m_t', m_t, sep='\n')
        # print('v_t', v_t, sep='\n')
        M_t = m_t / (1 - self.b1 ** self.t)
        V_t = v_t / (1 - self.b2 ** self.t)
        # print('M_t', M_t, sep='\n')
        # print('V_t', V_t, sep='\n')
        
        step = learning_rate * M_t / (np.sqrt(V_t) + self.e)
        
        res = input - step
        
        self.t += 1
        self.m_prev = m_t
        self.v_prev = v_t
        
        return res
    
def soft_results(data):
    # print(data)
    
    data[np.isnan(data)] = 0.00000001
    data[data == np.inf] = 999999
    data[data == -np.inf] = -99999
    # print('\n\n', data, '\n\n')
    # sleep(1)
    return data
    

def to_full(y, num_classes):
    """
    Returns array with 1 in class index position

    Parameters:
    -y: correct class index
    -num_classes: len of classes sequency
    """
    y_full = np.zeros(num_classes)
    y_full[y] = 1
    return y_full

class Layer:
    def recalculate(self):
        pass
    
    def get_results(self):
        pass
    
    def get_outs_number(self):
        pass
    
    def evaluate(self, y, is_last = True, learning_rate=0.00002, input_ = None, ce_type = 1):
        pass
    
    def setIn(self, in_):
        pass
    
    
class BatchNormalization(Layer):
    
    def __init__(self, in_:Dense|InputLayer, neurons_count, activation_func, optimizer1:Optimizer = BaseOprimizer(), optimizer2:Optimizer = BaseOprimizer()):
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
        
        self.outs=soft_results(h)
        
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
    
    
class Dense(Layer):
    def __init__(self, units, activation, optimizer1:Optimizer = BaseOprimizer(), optimizer2:Optimizer = BaseOprimizer(), in_:Layer = None, input_shape = None):
        self.__H_DIM = units
        if input_shape:
            self.prev_layer = InputLayer(input_shape)
            self.W = np.random.rand(self.prev_layer.get_outs_number(), self.__H_DIM)
        elif in_:
            self.prev_layer = in_
            self.W = np.random.rand(self.prev_layer.get_outs_number(), self.__H_DIM)
        else:
            self.W = np.array([])
        self.b = np.random.rand(1, self.__H_DIM)
        self.W = (self.W - 0.5) * 2 * np.sqrt(1/self.__H_DIM)     # new
        self.b = (self.b - 0.5) * 2 * np.sqrt(1/self.__H_DIM)     # new
        
        self.func = relu if activation == 'relu' else softmax if activation == 'softmax' else sigmoid if activation == 'sigmoid' else activation
        self.outs = None
        self.optimizer1 = optimizer1
        self.optimizer2 = optimizer2
        
    def setIn(self, in_):
        self.prev_layer = in_
        self.W = np.random.rand(self.prev_layer.get_outs_number(), self.__H_DIM)
   
    def recalculate(self):
        t = self.prev_layer.get_results() @ self.W + self.b
        h = self.func(t)
        self.t = t
        self.outs= soft_results(h)
        
    def get_results(self)  -> np.ndarray:
        return self.outs
        
    def get_outs_number(self) -> int:
        return self.__H_DIM

    def evaluate(self, y, is_last = True, learning_rate=0.00002, input_ = None, ce_type = 1):
        if is_last:
            # where is softmax???
            # y_full = to_full(y, self.__H_DIM)   # here Y params is a row matrix of wanted result class
            if ce_type in [1, 2] and self.func.__name__ in ['softmax', 'tanh', 'sigmoid']:
                if ce_type == 1:
                    if self.func.__name__ == 'softmax':
                        dE_dt = self.outs - y
                    elif self.func.__name__ == 'sigmoid':
                        dE_dt = self.outs - y
                    elif self.func.__name__ == 'tanh':
                        dE_dt = self.outs - y # 1 - y + self.outs - y/self.outs
                elif ce_type == 2:
                    if self.func.__name__ == 'softmax':
                        dE_dt = self.outs - y
                    elif self.func.__name__ == 'sigmoid':
                        dE_dt =(1-self.outs)*y
                    elif self.func.__name__ == 'tanh':
                        dE_dt = (1/self.outs-self.outs)*y # 1 - y + self.outs - y/self.outs
            elif ce_type in [3]:
                if ce_type == 3:
                    dE_dt = self.func(self.t, dif = True)*(self.outs - y)
            else:
                dE_dt = -y/self.outs * self.func(self.t, dif=True)
            # print(y, self.outs, dE_dt, sep='\n')
            # dE_dt = np.sum(dE_dt, axis=0, keepdims=True)/len(dE_dt)
            # print(dE_dt)
            dE_dW = self.prev_layer.get_results().T @ dE_dt
            dE_db = np.sum(dE_dt, axis=0, keepdims=True)
            dE_dh_prev = dE_dt @ self.W.T
        else:
            dE_dt = y * self.func(self.t, dif=True) # here Y params is a derivative matrix of current layer outs
            # print('y', y, 'prevouts', self.prev_layer.get_results(), '\n de_dt', dE_dt)
            dE_dW = self.prev_layer.get_results().T @ dE_dt
            dE_db = np.sum(dE_dt, axis=0, keepdims=True)
            dE_dh_prev = dE_dt @ self.W.T
            # print('dE_dh_prev', dE_dh_prev)
        # print('prev', self.__class__, is_last, self.W.size, self.b.size)
        self.W = self.optimizer1.calc(self.W, learning_rate, dE_dW)
        self.b = self.optimizer2.calc(self.b, learning_rate, dE_db)
        # print('new', self.__class__, is_last, self.W.size, self.b.size)
        if type(self.prev_layer) == InputLayer:
            return None
        return self.prev_layer.evaluate(dE_dh_prev, is_last=False, learning_rate=learning_rate, ce_type=ce_type)
      
    
class InputLayer:
    
    def __init__(self, inputs_size):
        self.X = None
        self.__H_DIM = inputs_size
    
    def set_input(self, inputs):
        self.X = np.asarray(inputs)
        
    def get_outs_number(self) -> int:
        return self.__H_DIM
    
    def get_results(self) -> np.ndarray:
        return self.X


class Sequential:
    
    def __init__(self, optimizer:Optimizer|str, layers:list[Layer] = [], type_='crossentropy', ALPHA = 0.0002 , class_number = 2):
        
        self.optimizer = optimizer
        
        self.layers: list[InputLayer | Dense] = []
        for layer in layers:
            self.add(layer)
        
        # print(self.layers)
        self.type_ = type_
        print(self.type_)
        self.learning_rate = ALPHA
        self.class_number = class_number
        
    
    def getOptimizer(self, optimizer):
        if type(optimizer) == str:
            if optimizer == 'adam':
                return Adam()
        else:
            if type(optimizer) == Adam:
                return Adam(optimizer.b1, optimizer.b2, optimizer.e)
        return BaseOprimizer()

        
    def add(self, layer:Dense):
        if len(self.layers) < 1:
            if type(layer.prev_layer) != InputLayer:
                raise Exception('Adding First Layer Error: incorrect layer input')
            layer.optimizer1 = self.getOptimizer(self.optimizer)
            layer.optimizer2 = self.getOptimizer(self.optimizer)
            self.layers.append(layer)
        else:
            layer.setIn(self.layers[-1])
            layer.optimizer1 = self.getOptimizer(self.optimizer)
            layer.optimizer2 = self.getOptimizer(self.optimizer)
            self.layers.append(layer)
            
        
    def fit(self, dataset, num_epochs, need_calculate_accuracy = False, need_calculate_loss = False, batch_size = 8):
        loss_arr = []
        accuracy_arr = []
        
        for ep in range(num_epochs):
            
            
            random.shuffle(dataset)
            for i in range(len(dataset) // batch_size):

                batch_x, batch_y = zip(*dataset[i*batch_size : i*batch_size+batch_size])
                # print(batch_y)
                x = np.concatenate(batch_x, axis=0)
                y = np.array(batch_y)
                if type(y) == np.ndarray:
                    # passed += 1
                    self.layers[0].prev_layer.set_input(x)
                    for layer in self.layers:
                        layer.recalculate()
                    
                    z=self.layers[-1].get_results()
                    

                    if self.type_ == 'crossentropy':                        
                        # Backward
                        self.layers[-1].evaluate(y, learning_rate=self.learning_rate, input_ = x, ce_type=1)
                    elif self.type_ == 'binary_crossentropy':
                        # Backward
                        self.layers[-1].evaluate(y, learning_rate=self.learning_rate, input_ = x, ce_type=2)
                    elif self.type_ == 'mean_squared_error':
                        # Backward
                        self.layers[-1].evaluate(y, learning_rate=self.learning_rate, input_ = x, ce_type=3)
                    elif self.type_ == 'test':
                        # Backward
                        self.layers[-1].evaluate(y, learning_rate=self.learning_rate, input_ = x, ce_type=4)
                    else:
                        raise Exception('Unknown model type')
                
            
            if need_calculate_loss:
                loss_arr.append(sparse_cross_entropy(z, y))
            if need_calculate_accuracy:
                accuracy_arr.append(self.calc_accuracy(dataset))
                
                    
        return loss_arr, accuracy_arr
                
    def predict(self, x):
        
        self.layers[0].prev_layer.set_input(x)
        for layer in self.layers:
            # print(layer.W)
            layer.recalculate()
        # print(self.layers[-1].t)
        z=self.layers[-1].get_results()
        return z
    
    def calc_accuracy(self, dataset):
        correct = 0
        passed = 0
        for x, y in dataset:
            if type(y) == np.ndarray:
                z = self.predict(x)
                if z.size > 1:
                    y_pred = np.argmax(z) # get position ?
                    if y_pred == np.argmax(y):
                        correct += 1
                else:
                    if np.max(z) >= 0.0 and np.max(y):
                        correct += 1
            else:
                passed += 1
        acc = correct / ( len(dataset) - passed )
        return acc

    def to_json(self):
        json_model = {}
        json_model["class_name"] = self.__class__.__name__
        config = {}
        config['name'] = json_model['class_name'].lower()
        layers = []
        denses = self.layers
        for i in range(len(denses)):
            dense = denses[i]

            if dense.prev_layer.__class__.__name__ == "InputLayer":
                input_layer = dense.prev_layer

                input_dense = {}
                input_dense["module"] = 'keras.layer'
                input_dense['class_name'] = input_layer.__class__.__name__
                d_conf = {}
                d_conf["batch_input_shape"] = [None, input_layer.get_outs_number()]
                input_dense["dtype"] = 'float32'
                input_dense["sparse"] = False
                input_dense["ragged"] = False
                input_dense["name"] = 'dense_input'
                input_dense['config'] = d_conf

                layers.append(input_dense)

            l_dense = {}
            l_dense["module"] = 'keras.layer'
            l_dense['class_name'] = dense.__class__.__name__
            d_conf = {}
            d_conf["batch_input_shape"] = [None, dense.prev_layer.get_outs_number()]

            name = 'dense' if i == 0 else f'dense_{i}'

            d_conf['name'] = name
            d_conf['trainable'] = True
            d_conf['dtype'] = 'float32'
            d_conf['units'] = dense.get_outs_number()
            d_conf['activation'] = dense.func.__name__
            d_conf['use_bias'] = True
            kernel_init = {}
            kernel_init["module"] = 'keras.initializers'
            kernel_init["class_name"] = 'RandomUniform'
            kernel_init["config"] = {}
            kernel_init["registered_name"] = None
            bias_init = {}
            bias_init["module"] = 'keras.initializers'
            bias_init["class_name"] = 'RandomUniform'
            bias_init["config"] = {}
            bias_init["registered_name"] = None
            d_conf['kernel_initializer'] = kernel_init
            d_conf['bias_initializer'] = bias_init
            d_conf["kernel_regularizer"] = None
            d_conf["bias_regularizer"] = None
            d_conf["activity_regularizer"] = None
            d_conf["kernel_constraint"] = None
            d_conf["bias_constraint"] = None

            l_dense['config'] = d_conf

            layers.append(l_dense)

        config["layers"] = layers
        json_model["config"] = config
        return json_model

    def save_weights(self):
        denses = self.layers
        denses_for_save = []

        for i in range(len(denses)):
            layer = {}
            dense = denses[i]
            name = 'dense' if i == 0 else f'dense_{i}'

            layer["name"] = name
            weights = []
            weights.append({'name': 'bias:0', 'numpy': dense.b[0]})
            weights.append({'name': 'kernel:0', 'numpy': dense.W})
            layer["weights"] = weights
            denses_for_save.append(layer)
        return denses_for_save

    def save(self, filename):
        with h5py.File(filename, "w") as hf:
            # Сохраняем архитектуру
            d = json.dumps(self.to_json())
            hf.attrs["model_architecture"] = d

            # Сохраняем веса
            for layer in self.save_weights():
                g = hf.create_group(layer['name'])
                for weight in layer['weights']:
                    weight_value = weight['numpy']
                    g.create_dataset(weight['name'], data=weight_value)


def load_model(filename):
    with h5py.File(filename, "r") as hf:
        model_architecture = hf.attrs["model_architecture"]
        model_architecture = json.loads(model_architecture)
        # print(model_architecture['config']['layers'][1])
        model = Sequential('adam',
                           [Dense(layer['config']['units'], layer['config']['activation'],
                                  input_shape=layer['config']['batch_input_shape'][1])
                            if i == 0 else Dense(layer['config']['units'], layer['config']['activation'])
                            for i, layer in enumerate(model_architecture['config']['layers'][1:])], ALPHA=0.001)
        for i, layer in enumerate(model.layers):
            name = 'dense' if i == 0 else f'dense_{i}'
            layer.b = hf[name]['bias:0'][...]
            layer.W = hf[name]['kernel:0'][...]
        print(model)
        return model


def save_layers_for_plot(model):
    denses = model.layers
    table_data = []
    for i in range(len(denses)):
        dense = denses[i]

        if dense.prev_layer.__class__.__name__ == "InputLayer":
            layer_input = []
            layer_output = []

            layer_input.append('dense_input')
            layer_input.append("input:")
            layer_input.append([None, dense.prev_layer.get_outs_number()])

            layer_output.append(dense.prev_layer.__class__.__name__)
            layer_output.append('output:')
            layer_output.append([None, dense.prev_layer.get_outs_number()])

            table_data.append(layer_input)
            table_data.append(layer_output)

        layer_input = []
        layer_output = []

        layer_input.append('dense' if i == 0 else f'dense_{i}')
        layer_input.append("input:")
        layer_input.append([None, dense.prev_layer.get_outs_number()])

        layer_output.append(dense.__class__.__name__)
        layer_output.append('output:')
        layer_output.append([None, dense.get_outs_number()])

        table_data.append(layer_input)
        table_data.append(layer_output)
    return table_data


from PIL import Image, ImageDraw, ImageFont


def plot_model(model, filename):
    image = Image.new("RGB", (0, 0), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    table_data = save_layers_for_plot(model)

    max_width = sum(
        [max([draw.textsize(str(row[i]), font=font)[0] + 10 for row in table_data]) for i in range(len(table_data[0]))])
    max_height = len(table_data) * 20 + ((len(table_data) // 2) - 1) * 40

    # Создаем новое изображение
    width, height = max_width + 20, max_height + 20
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    # max_len = 0
    start_x, all_x, prev_all_x = 0, 0, 0
    x, y = 0, 10
    cell_height = 20
    line_height = 20
    for i in range(0, len(table_data), 2):
        rows = table_data[i:i + 2]
        if not len(rows) == 0:
            column_widths = [max([draw.textsize(str(row[i]), font=font)[0] + 10 for row in rows]) for i in
                             range(len(rows[0]))]
            all_x = sum(column_widths)
            # max_len = all_x if all_x > max_len else max_len
            if start_x == 0:
                x = (max_width - all_x) / 2 + (width - max_width) / 2
            if not prev_all_x == 0:
                center = start_local_x + (prev_all_x / 2)
                draw.line(((center, y), (center, y + line_height)), fill="black", width=1)
                draw.polygon(
                    [(center - 7, y + line_height), (center + 7, y + line_height), (center, y + line_height * 2)],
                    fill='black')
                y += line_height * 2
                x = start_local_x + (prev_all_x - all_x) / 2
            start_local_x = x
            for row in rows:
                for i, cell in enumerate(row):
                    draw.rectangle([x, y, x + column_widths[i], y + cell_height], outline="black")
                    draw.text((x + 5, y + 5), str(cell), font=font, fill="black")
                    x += column_widths[i]
                y += cell_height
                x = start_local_x
            prev_all_x = all_x
    image.save(filename)
    # image.show()

import random
import numpy as np
from sklearn import datasets
iris = datasets.load_iris()



def tanh(t, dif=False):
    if dif:
        return 1. - tanh(t) ** 2
    return (np.exp(t) - np.exp(-t)) / (np.exp(t) + np.exp(-t))

def relu(t, dif=False):
    if dif:
        return (t >= 0).astype(float)
    return np.maximum(t, 0)

def linear(t, dif=False):
    if dif:
        return np.ones_like(t)
    return t

def backLinear(t, dif=False):
    if dif:
        return -1 *np.ones_like(t)
    return -t
def inverseProportion(t, dif=False):
    if dif:
        return -2/(t*3 + 0.1)
    return 1/(t**2 + 0.1)

def sin(t, dif=False):
    if dif:
        return np.cos(t)
    return np.sin(t)

def power3(t, dif=False):
    if dif:
        return 3*(t**2)
    return t**2

def cos(t, dif=False):
    if dif:
        return -np.sin(t)
    return np.cos(t)

def sigmoid(t, dif=False):
    if dif:
        return sigmoid(t)*(1-sigmoid(t))
    return 1/(1 + np.e **(-t))

def softmax(t, dif=False):
    # print(t)
    out = np.exp(t)
    if dif:
        return 1 / out
    # print(out)
    return out / np.sum(out, axis=1, keepdims=True)

def sparse_cross_entropy(z, y):
    try:
        return -np.log(z[0, y])
    except:
        return -np.log(z[0, 0])
        




    

if __name__ == '__main__':
    
    dataset = [(iris.data[i][None, ...], to_full(iris.target[i], 3)) for i in range(len(iris.target))]
    random.shuffle(dataset)
    # print(dataset[0])
    train = dataset[:len(dataset) - len(dataset)//15]
    test = dataset[-len(dataset)//15:]
    
    
    #code
    model2 = Sequential('adam', ALPHA=0.0001)
    model2.add(Dense(20, relu, input_shape=4))
    model2.add(Dense(10, relu))
    model2.add(Dense(3, softmax))
    loss_arr, accuracy_arr = model2.fit(train, 1000, need_calculate_loss=True, need_calculate_accuracy=True, batch_size=4)   
    print('test accuraccy', model2.calc_accuracy(train), model2.calc_accuracy(test))
    
    
    
    import matplotlib.pyplot as plt
    plt.plot(loss_arr)
    plt.plot(accuracy_arr)    
    plt.show()
  
