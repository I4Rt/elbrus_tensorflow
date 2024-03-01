from model.Sequential import Sequential
from model.layers.Dense import Dense
from model.layers.Flatten import Flatten
from model.layers.Conv2D import Conv2D

import json
import h5py


class LoadTools():

    @staticmethod
    def load_weights(model, filename):
        with h5py.File(filename, "r") as hf:
            for i, layer in enumerate(model.layers):
                name = 'dense' if i == 0 else f'dense_{i}'
                for value in list(hf['model_weights'][name][name]):
                    if value[0] == 'b':
                        layer.b = hf['model_weights'][name][name][value][...]
                    else:
                        layer.W = hf['model_weights'][name][name][value][...]
        return model

    @classmethod
    def create_layer(cls, layer, has_input):
        if layer['class_name'] == 'Dense':
            return_layer = Dense(layer['config']['units'], layer['config']['activation'],
                                 input_shape=layer['config']['batch_input_shape'][1]) if has_input \
                else Dense(layer['config']['units'], layer['config']['activation'])
        elif layer['class_name'] == 'Flatten':
            return_layer = Flatten()
        elif layer['class_name'] == 'Conv2D':
            print(layer['config']['batch_input_shape'][1])
            return_layer = Conv2D(tuple(layer['config']['kernel_size']), layer['config']['activation'],
                                 input_shape=tuple(layer['config']['batch_input_shape'][1])) if has_input \
                else Conv2D(layer['config']['kernel_size'], layer['config']['activation'])
        return return_layer

    @staticmethod
    def load_model(filename):
        with h5py.File(filename, "r") as hf:
            model_architecture = hf.attrs["model_architecture"]
            model_architecture = json.loads(model_architecture)
            # print(model_architecture['config']['layers'][1])
        # try:
        model = Sequential(model_architecture['config']['optimizer'],
                               [LoadTools.create_layer(layer, True if i == 0 else False)
                                for i, layer in enumerate(model_architecture['config']['layers'][1:])],
                               ALPHA=model_architecture['config']['alpha'])
        model = LoadTools.load_weights(model, filename)
        print(model)
        return model
        # except Exception as e:
        #     print('error in loading, maybe incorrect classname: ', e)
            # return -1


