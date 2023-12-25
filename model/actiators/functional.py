import numpy as np

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


def softMinusOneToPlusOne(y, dif=False):
    if dif:
        return 1/(1+np.sqrt(y**2))**2
    return y/(1+np.sqrt(y**2))

def softZeroToOne(y, dif=False):
    if dif:
        return 0.5/(1+np.sqrt(y**2))**2
    return 0.5+0.5*y/(1+np.sqrt(y**2))