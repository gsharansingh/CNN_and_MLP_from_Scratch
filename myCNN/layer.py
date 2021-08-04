import numpy as np

class Padding:
    def __init__(self, size = 0):
        self.size = size
    def __call__(self, data):
        img_height = data.shape[0]
        img_width = data.shape[1]
        zero_pad = np.zeros(((2*self.size)+img_height, (2*self.size)+img_width))
        padded_data = zero_pad[size:img_height+size+1][size:img_height+size+1] = data
        return padded_data

class Relu:
    def __init__(self):