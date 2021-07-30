import numpy as np
import myMLP.layers as layers

class Linear:
    def __init__(self, n_hidden = 1):
        self.X, self.W, self.b = layers.Input(), layers.Input(), layers.Input()
        self.n_hidden = n_hidden

    def __call__(self, X_, W_ = None, b_ = None):
        self.X.value = X_
        n_features = X_.shape[1]
        if W_:
            self.W.value = W_
        else:
            self.W.value = 2*np.random.random((n_features, self.n_hidden))-1
        if b_:
            self.b.value = b_
        else:
            self.b.value = 2*np.random.random((self.n_hidden, 1))-1

        return layers.Linear(self.X, self.W, self.b)
        

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
        first_layer = arg_layers.pop(0)
        for layer in arg_layers:
            pass
    
    def fit(self, X, y):
        first_layer(X)
