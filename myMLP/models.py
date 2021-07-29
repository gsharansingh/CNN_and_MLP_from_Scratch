import numpy as np
import myMLP.layers as layers

class Linear:
    def __init__(self, X, W = None, b = None):
        if W:
            self.W = W
        else:
            self.W = 2*np.random.random((X.shape[1], hidden_units))-1
        if b:
            self.b = b
        else:
            self.b = 2*np.random.random((hidden_units, 1))-1
        

class Sigmoid:
    def __init__(self, in_nodes = None):
        if in_nodes:
            self.obj = layers.Sigmoid(in_nodes)
    
    def __call__(self, in_nodes = None):
        if in_nodes:
            self.obj = layers.Sigmoid(in_nodes)
        if self.obj:
            return self.obj
        else:
            raise "Pass nodes through the layer"

class Sequential:
    def __init__(self, *arg_layers):
        for layer in arg_layers:
            pass
    
    def fit(self, X, y):
        feed_dict = {
            self.x = x,
            self.y = y
        }
        graph = layers.topological_sort(feed_dict)
