import numpy as np
from ../myMLP import layers

class Sigmoid:
    def __init__(self, in_nodes = None)
        if in_nodes:
            self.obj = layers.Sigmoid(in_nodes)
    
    def __call__(self, in_nodes = None)
        if in_nodes:
            self.obj = layers.Sigmoid(in_nodes)
        return self.obj

class Sequential:
    def __init__(self, *arg_layers):
        for layer in arg_layers:
