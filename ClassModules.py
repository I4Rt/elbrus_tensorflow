from __future__ import annotations
import numpy as np



def to_full(y, num_classes):
    """
    Returns array with 1 in class index position

    Parameters:
    -y: correct class index
    -num_classes: len of classes sequency
    """
    y_full = np.zeros((1, num_classes))
    y_full[0, y] = 1
    return y_full

class NeuronLayer:
    
    def __init__(self, in_:NeuronLayer|InputLayer, neurons_count, activation_func):
        self.prev_layer = in_
        
        self.__H_DIM = neurons_count
        self.W = np.random.rand(self.prev_layer.get_outs_number(), self.__H_DIM)
        self.b = np.random.rand(1, self.__H_DIM)
        self.func = activation_func
        self.outs = None
        
    def recalculate(self):
        t = self.prev_layer.get_results() @ self.W + self.b
        h = self.func(t)
        self.t = t
        self.outs=h
        
    def get_results(self)  -> np.ndarray:
        return self.outs
        
    def get_outs_number(self) -> int:
        return self.__H_DIM


    def evaluate(self, y, is_last = True, learning_rate=0.00002):
        
        if is_last:       # where is softmax???
            # y_full = to_full(y, self.__H_DIM)   # here Y params is a row matrix of wanted result class
            dE_dt = self.outs - y
            dE_dW = self.prev_layer.get_results().T @ dE_dt
            dE_db = np.sum(dE_dt, axis=0, keepdims=True)
            dE_dh_prev = dE_dt @ self.W.T

            self.W = self.W - learning_rate * dE_dW
            self.b = self.b - learning_rate * dE_db
            
            return self.prev_layer.evaluate(dE_dh_prev, is_last=False)
        else:
            if type(self.prev_layer) != InputLayer:
                dE_dt = y * self.func(self.t, dif=True) # here Y params is a derivative matrix of current layer outs
                dE_dW = self.prev_layer.get_results().T @ dE_dt
                dE_db = dE_dt
                dE_dh_prev = dE_dt @ self.W.T
                return self.prev_layer.evaluate(dE_dh_prev, is_last=False)
            else:
                return None
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
    
    
    
if __name__ == '__main__':
    import random
    import numpy as np
    from sklearn import datasets
    iris = datasets.load_iris()

    INPUT_DIM = 4
    OUT_DIM = 3
    H_DIM = 10

    

    def relu(t, dif=False):
        if dif:
            return (t >= 0).astype(float)
        return np.maximum(t, 0)
    def sigmoid(t, dif=False):
        if dif:
            return 1/(1 + np.e **(-t))
        return  (np.e **(-t)) / ((1 + np.e **(-t)) ** 2)
    
    def softmax(t):
        out = np.exp(t)
        return out / np.sum(out)

    def sparse_cross_entropy(z, y):
        return -np.log(z[0, y])

    def to_full(y, num_classes):
        y_full = np.zeros((1, num_classes))
        y_full[0, y] = 1
        return y_full

    
        

    dataset = [(iris.data[i][None, ...], iris.target[i]) for i in range(len(iris.target))]

    input_layer = InputLayer(4)
    first_layer = NeuronLayer(input_layer, 10, relu)
    second_layer = NeuronLayer(first_layer, 3, softmax)

    def predict(x):
        input_layer.set_input(x)
        first_layer.recalculate()
        second_layer.recalculate()
        
        z=second_layer.get_results()
        return z

    def calc_accuracy():
        correct = 0
        for x, y in dataset:
            z = predict(x)
            y_pred = np.argmax(z)
            if y_pred == y:
                correct += 1
        acc = correct / len(dataset)
        return acc

    

    ALPHA = 0.0002
    NUM_EPOCHS = 400
    # BATCH_SIZE = 50

    loss_arr = []
    accuracy_arr = []
    for ep in range(NUM_EPOCHS):
        random.shuffle(dataset)
        for i in range(len(dataset)):
            x, y = dataset[i]

            input_layer.set_input(x)
            first_layer.recalculate()
            second_layer.recalculate()
            
            z=second_layer.get_results()
            E = sparse_cross_entropy(z, y)

            # Backward
            second_layer.evaluate(to_full(y, 3), learning_rate=ALPHA)


            accuracy_arr.append(calc_accuracy())
            loss_arr.append(E)

    

    accuracy = calc_accuracy()
    print("Accuracy:", accuracy)

    import matplotlib.pyplot as plt
    plt.plot(loss_arr)
    plt.plot(accuracy_arr)    
    plt.show()