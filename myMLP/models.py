import numpy as np
import myMLP.layers as layers

class Linear:
    def __init__(self, n_hidden = 1):
        self.X, self.W, self.b = layers.Input(), layers.Input(), layers.Input()
        self.n_hidden = n_hidden
        self.input_nodes = [self.X, self.W, self.b]

    def __call__(self, X_, W_ = None, b_ = None):
        self.X.value = X_.astype(np.float32)
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

class Softmax:
    def __init__(self, in_nodes = None):
        self.input_nodes = None
        if in_nodes:
            self.obj = layers.Softmax(in_nodes)
    
    def __call__(self, in_nodes = None):
        if in_nodes:
            self.obj = layers.Softmax(in_nodes)
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

class CrossEntropy:
    def __init__(self):
        self.input_nodes = None
        self.Y = layers.Input()
    def __call__(self, y_hat, y):
        self.Y.value = y
        return layers.CrossEntropy(y_hat, self.Y)

class Sequential:
    def __init__(self, *arg_layers):
        self.arg_layers = arg_layers
        self.input_nodes = []
    
    def fit(self, X, y):
        x = self.arg_layers[0](X)
        for layer in self.arg_layers[1:-1]:
            x = layer(x)
        self.arg_layers[-1](x, y)
        for layer in self.arg_layers:
            if layer.input_nodes:
                self.input_nodes += layer.input_nodes

        self.graph = layers.topological_sort_list(self.input_nodes)
        layers.forward_and_backward(self.graph)

    def predict(self, x):
        self.graph[0].value = x
        for n in self.graph[:-1]:
            n.forward()

        return self.graph[-1].value

def batches(batch_size, features, labels):
    assert len(features) == len(labels)
    output_batches = []
    sample_size = len(features)
    for start_i in range(0, sample_size, batch_size):
        end_i = start_i + batch_size
        batch = [features[start_i:end_i], labels[start_i:end_i]]
        output_batches.append(batch)
    return output_batches

class Flatten:
    def __init__(self):
        self.data_2d = []
    def __call__(self, data):
        for i in data:
            self.data_2d.append(i.reshape(1, -1))
        return np.array(self.data_2d)
