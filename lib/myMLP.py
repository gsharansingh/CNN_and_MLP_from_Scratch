import numpy as np

class Node(object):
    def __init__(self, in_nodes = []):
        self.in_nodes = in_nodes
        self.value = None
        self.out_nodes = []
        for n in in_nodes:
            n.out_nodes.append(self)

class Input(Node):
    def __init__(self):
        Node.__init__(self)

    def forward(self):
        pass

class Linear(Node):
    def __init__(self, X, W, b):
        Node.__init__(self, [X, W, b])

    def forward(self):
        X = self.in_nodes[0].value
        W = self.in_nodes[1].value
        b = self.in_nodes[2].value
        self.value = np.matmul(X, W.T) + b

class Sigmoid(Node):
    def __init__(self, a):
        Node.__init__(self, [a])

    def forward(self):
        a = self.in_nodes[0].value
        self.value = self._sigmoid(a)

    def _sigmoid(self, x):
        return 1.0/(1.0 + np.exp(-x))