from model.Sequential import Sequential
from model.layers.Dense import Dense

import json
import h5py

class LoadTools():

    @staticmethod
    def load_weights(model, filename):
        with h5py.File(filename, "r") as hf:
            for i, layer in enumerate(model.layers):
                name = 'dense' if i == 0 else f'dense_{i}'
                layer.b = hf['model_weights'][name][name]['bias:0'][...]
                layer.W = hf['model_weights'][name][name]['kernel:0'][...]
        return model

    @staticmethod
    def load_model(filename):
        with h5py.File(filename, "r") as hf:
            model_architecture = hf.attrs["model_architecture"]
            model_architecture = json.loads(model_architecture)
            # print(model_architecture['config']['layers'][1])
        model = Sequential('adam',
                           [Dense(layer['config']['units'], layer['config']['activation'],
                                  input_shape=layer['config']['batch_input_shape'][1]) if i == 0
                            else Dense(layer['config']['units'], layer['config']['activation'])
                            for i, layer in enumerate(model_architecture['config']['layers'][1:])], ALPHA=0.001)
        model = LoadTools.load_weights(model, filename)
        print(model)
        return model