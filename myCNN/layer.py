import numpy as np

class Padding:
    def __init__(self, size = 1):
        self.size = size
    def __call__(self, data):
        img_height = data.shape[0]
        img_width = data.shape[1]
        zero_pad = np.zeros(((2*self.size)+img_height, (2*self.size)+img_width))
        zero_pad[self.size:img_height+self.size, self.size:img_height+self.size] = data
        return zero_pad

class Relu:
    def __call__(self, data):
        # element-wise maximum comparison
        return np.maximum(data, 0)

class MaxPool:
    def __init__(self, pool_size = 2):
        self.pool_size = pool_size
    def __call__(self, data):
        num_img = data.shape[0]
        img_height = data.shape[1]
        img_width = data.shape[2]
        new_img_height = int(img_height/self.pool_size)
        new_img_width = int(img_width/self.pool_size)
        pooled_data = np.zeros((num_img, new_img_height, new_img_width)) #creating an array to store pooled/sampled data

        #selecting the porition of the rectified data
        for num in range(num_img):
            for i in range (new_img_height):
                h_index = self.pool_size*i
                for j in range (new_img_width):
                    w_index = self.pool_size*j
                    # storing maximum value
                    pooled_data[num][i][j] = np.max(data[num][h_index: h_index+self.pool_size, w_index: w_index+self.pool_size])
        return pooled_data

class ConvLayer:
    def __init__(self, stride = 1, kernels = np.ones((1, 3, 3))):
        self.stride = stride
        self.kernels = kernels
        self.num_kernels = kernels.shape[0]
        self.kernel_size = kernels.shape[1]
    def __call__(self, data):
        img_height = data.shape[0]
        img_width = data.shape[1]
        conv_img_height = int((img_height-self.kernel_size)/self.stride)+1
        conv_img_width = int((img_width-self.kernel_size)/self.stride)+1
        conv_data = np.zeros((self.num_kernels, conv_img_height, conv_img_width))
        #Convolve image with filter pixel by pixel
        for num in range(self.num_kernels):
            for i in range(0, conv_img_height, self.stride):
                for j in range(0, conv_img_width, self.stride):
                    temp = self.kernels[num]*data[i:(i+self.kernel_size), j:(j+self.kernel_size)]
                    conv_data[num][i][j] = np.sum(temp)
        return conv_data

class Normalize:
    def __init__(self, type = 'binary'):
        self.type = type
    def __call__(self, data):
        if (self.type == 'binary'):
            return data/255.0