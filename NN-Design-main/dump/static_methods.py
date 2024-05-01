import numpy as np
import math



@staticmethod
def logsigmoid(n):
    return 1 / (1 + np.exp(-n))

@staticmethod
def logsigmoid_stable(n):
    n = np.clip(n, -100, 100)
    return 1 / (1 + np.exp(-n))

@staticmethod
def logsigmoid_der(n):
    return (1 - 1 / (1 + np.exp(-n))) * 1 / (1 + np.exp(-n))

@staticmethod
def purelin(n):
    return n

@staticmethod
def purelin_der(n):
    return np.array([1]).reshape(n.shape)

@staticmethod
def lin_delta(a, d=None, w=None):
    na, ma = a.shape
    if d is None and w is None:
        return -np.kron(np.ones((1, ma)), np.eye(na))
    else:
        return np.dot(w.T, d)

@staticmethod
def log_delta(a, d=None, w=None):
    s1, _ = a.shape
    if d is None and w is None:
        return -np.kron((1 - a) * a, np.ones((1, s1))) * np.kron(np.ones((1, s1)), np.eye(s1))
    else:
        return (1 - a) * a * np.dot(w.T, d)

@staticmethod
def tan_delta(a, d=None, w=None):
    s1, _ = a.shape
    if d is None and w is None:
        return -np.kron(1 - a * a, np.ones((1, s1))) * np.kron(np.ones((1, s1)), np.eye(s1))
    else:
        return (1 - a * a) * np.dot(w.T, d)

@staticmethod
def marq(p, d):
    s, _ = d.shape
    r, _ = p.shape
    return np.kron(p.T, np.ones((1, s))) * np.kron(np.ones((1, r)), d.T)

@staticmethod
def compet(n, axis=None):
    if axis is not None:
        max_idx = np.argmax(n, axis=axis)
        out = np.zeros(n.shape)
        for i in range(out.shape[1]):
            out[max_idx[i], i] = 1
        return out
    else:
        max_idx = np.argmax(n)
        out = np.zeros(n.shape)
        out[max_idx] = 1
        return out

@staticmethod
def poslin(n):
    return n * (n > 0)

@staticmethod
def hardlim(x):
    if x < 0:
        return 0
    else:
        return 1

@staticmethod
def hardlims(x):
    if x < 0:
        return -1
    else:
        return 1

@staticmethod
def satlin(x):
    if x < 0:
        return 0
    elif x < 1:
        return x
    else:
        return 1

@staticmethod
def satlins(x):
    if x < -1:
        return 0
    elif x < 1:
        return x
    else:
        return 1

@staticmethod
def logsig(x):
    return 1 / (1 + math.e ** (-x))

@staticmethod
def tansig(x):
    return 2 / (1 + math.e ** (-2 * x)) - 1

def nndtansig(self, x):
    a = tansig(x)
