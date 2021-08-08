import numpy as np

class Node(object):
    def __init__(self, in_nodes = []):
        self.in_nodes = in_nodes
        self.value = None
        self.gradients = {}
        self.out_nodes = []
        for n in in_nodes:
            n.out_nodes.append(self)

    def forward(self):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError

class Input(Node):
    def __init__(self):
        Node.__init__(self)

    def forward(self):
        pass

    def backward(self):
        self.gradients[self] = 0
        for n in self.out_nodes:
            grad_value = n.gradients[self]
            self.gradients[self] += grad_value * 1

class Linear(Node):
    def __init__(self, X, W, b):
        Node.__init__(self, [X, W, b])

    def forward(self):
        X = self.in_nodes[0].value
        W = self.in_nodes[1].value
        b = self.in_nodes[2].value
        self.value = np.matmul(X, W) + b

    def backward(self):
        self.gradients = {n: np.zeros_like(n.value) for n in self.in_nodes}
        for n in self.out_nodes:
            grad_value = n.gradients[self]
            self.gradients[self.in_nodes[0]] += np.dot(grad_value, self.in_nodes[1].value.T)
            self.gradients[self.in_nodes[1]] += np.dot(self.in_nodes[0].value.T, grad_value)
            self.gradients[self.in_nodes[2]] += np.sum(grad_value)#, axis=0, keepdims=False)

class Sigmoid(Node):
    def __init__(self, z):
        Node.__init__(self, [z])

    def forward(self):
        z = self.in_nodes[0].value
        self.value = self._sigmoid(z)

    def _sigmoid(self, x):
        return 1.0/(1.0 + np.exp(-x))

    def backward(self):
        self.gradients = {n: np.zeros_like(n.value) for n in self.in_nodes}
        for n in self.out_nodes:
            grad_value = n.gradients[self]
            sigmoid = self.value
            self.gradients[self.in_nodes[0]] += sigmoid * (1 - sigmoid) * grad_value

class Softmax(Node):
    def __init__(self, z):
        Node.__init__(self, [z])

    def forward(self):
        z = self.in_nodes[0].value
        self.value = self._softmax(z)
        self.d_softmax(z)

    def _softmax(self, x):
        exps = np.exp(x - x.max())
        return exps/np.sum(exps)

    def d_softmax(self, x):
        dx_ds = np.diag(x) - np.dot(x, x.T)
        self.dx = dx_ds.sum(axis=0).reshape(-1,1)

    def backward(self):
        self.gradients = {n: np.zeros_like(n.value) for n in self.in_nodes}
        for n in self.out_nodes:
            grad_value = n.gradients[self]
            softmax = self.value
            self.gradients[self.in_nodes[0]] += self.dx * grad_value

class MSE(Node):
    def __init__(self, y_hat, y):
        Node.__init__(self, [y_hat, y])

    def forward(self):
        y_hat = self.in_nodes[0].value
        y = self.in_nodes[1].value
        self.m = self.in_nodes[0].value.shape[1]
        self.diff = y_hat - y
        self.value = (1/2) * np.mean(np.square(self.diff), axis = 0)

    def backward(self):
        self.gradients[self.in_nodes[0]] = (self.diff / self.m)
        self.gradients[self.in_nodes[1]] = -(self.diff / self.m)

class CrossEntropy(Node):
    def __init__(self, y_hat, y):
        Node.__init__(self, [y_hat, y])

    def forward(self):
        y_hat = self.in_nodes[0].value.reshape(-1, 1)
        y = self.in_nodes[1].value.reshape(-1, 1).astype(np.float32)
        self.m = self.in_nodes[0].value.shape[0]
        log_likelihood = -np.log(y_hat, y)
        self.value = np.sum(log_likelihood) / m

    def backward(self):
        y_hat = self.in_nodes[0].value.reshape(-1, 1)
        y = self.in_nodes[1].value.reshape(-1, 1)
        y_hat[range(self.m), y] -= 1
        self.gradients[self.in_nodes[0]] = (y_hat/self.m)
        self.gradients[self.in_nodes[1]] = -(y_hat/self.m)

def topological_sort(feed_dict):
    input_nodes = [n for n in feed_dict.keys()]
    G = {}
    nodes = [n for n in input_nodes]
    while len(nodes) > 0:
        n = nodes.pop(0)
        if n not in G:
            G[n] = {'in': set(), 'out': set()}
        for m in n.out_nodes:
            if m not in G:
                G[m] = {'in': set(), 'out': set()}
            G[n]['out'].add(m)
            G[m]['in'].add(n)
            nodes.append(m)

    L = []
    S = set(input_nodes)
    while len(S) > 0:
        n = S.pop()

        if isinstance(n, Input):
            n.value = feed_dict[n]

        L.append(n)
        for m in n.out_nodes:
            G[n]['out'].remove(m)
            G[m]['in'].remove(n)
            # if no other incoming edges add to S
            if len(G[m]['in']) == 0:
                S.add(m)
    return L

def topological_sort_list(input_nodes):
    G = {}
    nodes = [n for n in input_nodes]
    while len(nodes) > 0:
        n = nodes.pop(0)
        if n not in G:
            G[n] = {'in': set(), 'out': set()}
        for m in n.out_nodes:
            if m not in G:
                G[m] = {'in': set(), 'out': set()}
            G[n]['out'].add(m)
            G[m]['in'].add(n)
            nodes.append(m)

    L = []
    S = set(input_nodes)
    while len(S) > 0:
        n = S.pop()

        L.append(n)
        for m in n.out_nodes:
            G[n]['out'].remove(m)
            G[m]['in'].remove(n)
            # if no other incoming edges add to S
            if len(G[m]['in']) == 0:
                S.add(m)
    return L

def forward_and_backward(graph):
    # Forward pass
    for n in graph:
        n.forward()

    # Backward pass
    for n in graph[::-1]:
        n.backward()