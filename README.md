# CNN_and_MLP_from_Scratch
Image Detection Algorithm is created by using Convolutional Neural Networks (CNN) and Multilayer Perceptron with Backpropagation Algorithms from Scratch
## Table of contents
* [General info](#general-info)
* [Technologies](#theoratical-concept-of-the-algorithm)
  * [Reducing the Image Size](#reducing-the-image-size)
  * [Extracting the features of Image using Convolutional Neural Network](#extracting-the-features-of-image-using-convolutional-neural-network)
* [Setup](#setup)
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
