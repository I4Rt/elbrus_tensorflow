import numpy as np
import random
from time import time

from model.layers.Layer import Layer
from model.layers.Dense import Dense
from model.layers.InputLayer import InputLayer

from model.optimizers.Optimizer import Optimizer
from model.optimizers.SGD import SGD
from model.optimizers.Adam import Adam

from model.tools.CrossEntropyTools import CrossEntropyTools
from model.tools.ProgressBar import ProgressBar

from model.tools.SaveTools import SaveTools
from model.tools import PlotTools


class Sequential:

    def __init__(self, optimizer:Optimizer|str, layers:list[Layer] = [], type_='crossentropy', ALPHA = 0.0002 ,  class_number = 2):

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
        return SGD()


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
        b = None
        # test_dataset = dataset[-len(dataset)//3::1]
        # dataset = dataset[:len(dataset)*2//3]

        for ep in range(num_epochs):
            ProgressBar.printProgressBar(ep + 1, num_epochs, prefix = 'Progress:', suffix = f'Complete | {round(time() - b, 3) if b else "?"} sec/era', length = 50)
            b = time()
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
                        if self.layers[-1].get_outs_number() != 1:
                            raise Exception('Wrong last layer outs size for BCE')
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
                loss_arr.append(CrossEntropyTools.sparse_cross_entropy(z, y))
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
                # print('size', z.size)
                # print('max_y', np.max(y))
                # print('max_z', np.max(z))
                if z.size > 1:
                    y_pred = np.argmax(z) # get position ?
                    if y_pred == np.argmax(y):
                        correct += 1
                else:
                    if np.max(y):
                        if np.max(z) >= 0.5:
                            correct += 1
                    else:
                        if np.max(z) < 0.5:
                            correct += 1
            else:
                passed += 1
        acc = correct / ( len(dataset) - passed )
        return acc

    def save(self, filepath):
        SaveTools.save(self, filepath)

    def to_json(self):
        return SaveTools.to_json(self)

    def save_weights(self):
        return SaveTools.save_weights(self)

