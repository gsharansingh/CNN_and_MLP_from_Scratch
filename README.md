# CNN_and_MLP_from_Scratch
Image Detection Algorithm is created by using Convolutional Neural Networks (CNN) and Multilayer Perceptron with Backpropagation Algorithms from Scratch

## Table of contents
* [General info](#general-info)
* [Technologies](#theoratical-concept-of-the-algorithm)
  * [Reducing the Image Size](#reducing-the-image-size)
  * [Extracting the features of Image using Convolutional Neural Network](#extracting-the-features-of-image-using-convolutional-neural-network)
* [](#setup)

### General info
This project is implementing the concept of Convolutional Neural Networks and Multilayer Perceptron to detect the number in the image (the number should be ranging between 1 to 5). The algorithm is built from scratch by using the theoretical knowledge of the Convolutional Neural Networks and Multilayer Perceptron. This is not a customized algorithm, which means it may not give good results. 

## Theoratical concept of the algorithm

###	Reducing the Image Size
An Image have different width and dimensions. So, before applying CNN, it is required to first make the image has same dimensions as of the images that were used to build the model. The steps are below:

1.	First, the Image is cropped to get the image of size ratio 1:1
This is done by seeing which dimension (i.e. height or width) of the size is smaller. The larger dimension is cropped from both sides to make it of the size as of smaller dimension (e.g. there is an image of size 400x600. So, here the larger dimension 600 (width) will get cropped to give an output image of size 400x600)

2.	The size of image is reduced, 16x16, by finding the normal value of block of multiple pixels and assigning that normal value to single pixel.
Since we need an image of size 16x16, because our model was trained by using these size images, we first find the reduction factor, which can be obtained by dividing the image dimension (height or width) obtained in Step 1 with 16 (image dimension used to train model). Then, the normal value of block of multiple pixels with size of reduction factor is found, which is then assigned to a single pixel. This gives an image of size 16x16.

3.	The reduced sized image is then converted into Black and White image by thresholding property.

Since, the multilayer perceptron model works best with binary values, a threshold function is used to get two values from the grayscale image. This is achieved by using “if” statement. Each value represents either white or black color in the image.

4.	Since the images used to train model has black background, the test image if have white background will get converted into black background. This is achieved by using “if” statement.

###	Extracting the features of Image using Convolutional Neural Network
The algorithm for CNN is built from the information gathered in [1] and [2].

The feature extraction can be done by using five steps:
1.	**Padding:** It is of two types – one in which convolved feature is reduced in dimensionality (This can be achieved by giving zero_padding_size = 0). Other type keeps the original size (zero_padding_size = 1) or can be used to increase the dimensionality (zero_padding_size > 1).

<h1 align="center">
<img src="raw%20images/zero_pad.png" width="200">
 </h1>
 
2.	**Filtering for feature extraction:** Filtering/Convolution is a process of using a filter on each pixel of image by moving it bit by bit. The extracted features depend upon the selection of filter/kernel.

<h1 align="center">
<img src="raw%20images/filter.gif" width="300">
</h1>
 
3.	**ReLU:** It is an activation function, used to remove negative values after filtering. It replaces negative values with zeros.

<h1 align="center">
<img src="raw%20images/relu.png" width="300">
</h1>
 
4.	**Max Pooling:** It is done by getting the maximum value from block image pixels (In the algorithm, we use pooling size = 2, which means it is taking maximum value from the image block of size 2x2.

<h1 align="center">
<img src="raw%20images/Maxpool.png" width="300">
</h1>
 
5.	The filtered output is 2D array, it is flattened for purpose of feeding it to Multilayer Perceptron.
 
<h1 align="center">
<img src="raw%20images/flatten.png" width="300">
</h1>
 
## Multilayer perceptron (Forward Propagation)
The image below shows how a multilayer perceptron looks like:

<h1 align="center">
<img src="raw%20images/mlp_fw.png" width="500">
</h1>

As it can be seen from the image above that a multilayer perceptron have an input layer, 1 or more hidden layers and 1 output layer. This is implemented by using the concepts presented in [3].

* Input Layer: The input data is provided on this layer. We normalized the data to either have value 0 or 1. So, that means we are giving 0’s and 1’s at input layer
*	Hidden Layer: A multilayer perceptron can have multiple hidden layers. But in our model, we are using just one hidden layer.
*	Output Layer: It gives the output depending upon distribution of weights

Operations at different layers:

<img src="raw%20images/forward.png" width="500">

## Multilayer perceptron (Back-Propagation)
It is used to provide the model with the ability of learning from error by using derivation and chain rule. The information is gathered from [3] and [4].

The algorithm works like:

<img src="raw%20images/backward.png" width="500">
