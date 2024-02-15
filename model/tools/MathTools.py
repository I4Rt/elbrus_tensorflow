import numpy as np

class MathTools:
    
    @staticmethod
    def soft_results(data):
        data[np.isnan(data)] = 0.00000001
        data[data == np.inf] = 999999
        data[data == -np.inf] = -99999
        return data

    @staticmethod
    def to_categorical(x, num_classes=None):
        x = np.array(x, dtype="int64")
        input_shape = x.shape

        # Shrink the last dimension if the shape is (..., 1).
        if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
            input_shape = tuple(input_shape[:-1])

        x = x.reshape(-1)
        if not num_classes:
            num_classes = np.max(x) + 1
        batch_size = x.shape[0]
        categorical = np.zeros((batch_size, num_classes))
        categorical[np.arange(batch_size), x] = 1
        output_shape = input_shape + (num_classes,)
        categorical = np.reshape(categorical, output_shape)
        return categorical
    