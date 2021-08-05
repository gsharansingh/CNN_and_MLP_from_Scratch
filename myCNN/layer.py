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
    def __call__(self, data):
        # element-wise maximum comparison
        return np.maximum(data, 0)

class MaxPool:
    def __init__(self, pool_size = 2):
        self.pool_size = pool_size
    def __call__(self, data):
        img_height = data.shape[0]
        img_width = data.shape[1]
        new_img_height = int(img_height/self.pool_size)
        new_img_width = int(img_width/self.pool_size)
        pooled_data = np.zeros((new_img_height, new_img_width)) #creating an array to store pooled/sampled data

        #selecting the porition of the rectified data
        for i in range (new_img_height):
            h_index = self.pool_size*i
            for j in range (new_img_width):
                w_index = self.pool_size*j
                # storing maximum value
                pooled_data[i][j] = np.max(data[h_index: h_index+self.pool_size, w_index: w_index+self.pool_size])
        return pooled_data