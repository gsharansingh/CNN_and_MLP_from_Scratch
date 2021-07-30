import numpy as np
import myMLP.layers as layers

class Linear:
    def __init__(self, n_hidden = 1):
        self.X, self.W, self.b = layers.Input(), layers.Input(), layers.Input()
        self.n_hidden = n_hidden
        self.input_nodes = [self.X, self.W, self.b]

    def __call__(self, X_, W_ = None, b_ = None):
        self.X.value = X_
        print(X_.shape)
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
        self.input_nodes = None
        if in_nodes:
            self.obj = layers.Sigmoid(in_nodes)
    
    def __call__(self, in_nodes = None):
        if in_nodes:
            self.obj = layers.Sigmoid(in_nodes)
        if self.obj:
            return self.obj
        else:
            raise "Pass nodes through the layer"

class MSE:
    def __init__(self):
        self.input_nodes = None
        self.Y = layers.Input()
    def __call__(self, y_hat, y):
        self.Y.value = y
        return layers.MSE(y_hat, self.Y)

class Sequential:
    def __init__(self, *arg_layers):
        self.arg_layers = arg_layers
        self.input_nodes = []
    
    def fit(self, X, y):
        print(self.arg_layers[0])
        x = self.arg_layers[0](X)
        for layer in self.arg_layers[1:-1]:
            x = layer(x)
        self.arg_layers[-1](x, y)
        for layer in self.arg_layers:
            if layer.input_nodes:
                self.input_nodes += layer.input_nodes

        graph = layers.topological_sort_list(self.input_nodes)
        layers.forward_and_backward(graph)
