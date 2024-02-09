import numpy as np
from numpy.lib.stride_tricks import as_strided


class ConvTools:

    @staticmethod
    def max_pooling_2D(kernel_size, stride, array):
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if hasattr(kernel_size, '__len__') and not len(kernel_size) == 2:
            print(f"`kernel_size` should have two elements. Received: {kernel_size}.")
            return -1
        output_shape = ((array.shape[0] - kernel_size[0]) // stride + 1,
                        (array.shape[1] - kernel_size[1]) // stride + 1)
        shape = (output_shape[0], output_shape[1], kernel_size[0], kernel_size[1])
        strides = (stride * array.strides[0], stride * array.strides[1], array.strides[0], array.strides[1])
        strided_array = as_strided(array, shape, strides)
        return strided_array.max(axis=(2, 3))

    @staticmethod
    def zero_padding_2D(padding, a):
        if isinstance(padding, int):
            padding = ((padding, padding), (padding, padding))
        if hasattr(padding, '__len__') and not len(padding) == 2:
            print(f"`padding` should have two elements. Received: {padding}.")
            return -1
        a_shape = np.shape(a)
        a_size = np.size(a_shape)
        if not a_size == 4:
            a = np.reshape(a, tuple(np.ones(4 - np.size(a_shape), dtype=int)) + a_shape)
        a_shape = np.shape(a)
        elem_size = a_shape[-1]

        for i in range(a_shape[0]):
            a_i = a[i]
            a_vs = np.vstack((np.zeros((padding[1][0], elem_size)), a_i[0], np.zeros((padding[1][1], elem_size))))
            a_i = np.reshape(a_vs, tuple(np.ones(4 - np.size(np.shape(a_vs)), dtype=int)) + np.shape(a_vs))
            a_vs = np.vstack((np.zeros((tuple([padding[0][0]]) + np.shape(a_i)[2:])), a_i[0],
                              np.zeros((tuple([padding[0][1]]) + np.shape(a_i)[2:]))))
            a_i = np.reshape(a_vs, tuple(np.ones(4 - np.size(np.shape(a_vs)), dtype=int)) + np.shape(a_vs))
            if i == 0:
                a_res = np.reshape(np.vstack(a_i),
                                   tuple(np.ones(4 - np.size(np.shape(np.vstack(a_i))), dtype=int)) + np.shape(
                                       np.vstack(a_i)))
            else:
                a_res = np.reshape(np.vstack((a_res, a_i)),
                                   tuple(np.ones(4 - np.size(np.shape(np.vstack((a_res, a_i)))), dtype=int)) + np.shape(
                                       np.vstack((a_res, a_i))))
        return a_res