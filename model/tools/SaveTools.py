import h5py
import json
# from model.Sequential import Sequential


class SaveTools:

    @classmethod
    def layer_conf(cls, dense, count):
        d_conf = {}
        d_conf["batch_input_shape"] = [None, dense.prev_layer.get_outs_number()]

        name = 'dense' if count == 0 else f'dense_{count}'

        d_conf['name'] = name
        d_conf['trainable'] = True
        d_conf['dtype'] = 'float32'
        if dense.__class__.__name__ == 'Conv2D':
            d_conf['kernel_size'] = dense.b.shape
        d_conf['units'] = dense.get_outs_number()
        if not dense.__class__.__name__ == 'Flatten':
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

        return d_conf


    @classmethod
    def to_json(cls, model):
        json_model = {}
        json_model["class_name"] = model.__class__.__name__
        config = {}
        config['name'] = json_model['class_name'].lower()
        config['optimizer'] = model.optimizer
        config['alpha'] = model.learning_rate
        layers = []
        denses = model.layers
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
            d_conf = cls.layer_conf(dense, i)

            l_dense['config'] = d_conf

            layers.append(l_dense)

        config["layers"] = layers
        json_model["config"] = config
        return json_model

    @classmethod
    def save_weights(cls, model):
        denses = model.layers
        denses_for_save = []

        for i in range(len(denses)):
            layer = {}
            dense = denses[i]
            name = 'dense' if i == 0 else f'dense_{i}'

            layer["name"] = name
            weights = []
            if hasattr(dense, 'b'):
                weights.append({'name': 'bias:0', 'numpy': dense.b})
            if hasattr(dense, 'W'):
                weights.append({'name': 'kernel:0', 'numpy': dense.W})
            layer["weights"] = weights
            denses_for_save.append(layer)
        return denses_for_save

    @classmethod
    def save(cls, model, filename):
        print(model)
        with h5py.File(filename, "w") as hf:
            # Сохраняем архитектуру
            d = json.dumps(cls.to_json(model))
            hf.attrs["model_architecture"] = d
            hf.attrs['backend'] = 'elbrus_tensorflow'
            model_weights = hf.create_group('model_weights')
            # Сохраняем веса
            for layer in cls.save_weights(model):
                layer_name = model_weights.create_group(layer['name'])
                group = layer_name.create_group(layer['name'])
                for weight in layer['weights']:
                    weight_value = weight['numpy']
                    group.create_dataset(weight['name'], data=weight_value)
                # layer_name.create_dataset(layer['name'], data=group)
                # model_weights.create_dataset(layer['name'], data=group)

