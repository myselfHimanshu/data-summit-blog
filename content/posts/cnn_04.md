+++
draft = true
date = 2020-04-12T22:22:22+05:30
title = "Convolutional Neural Network Part 4 : Architectural Design Basics"
slug = ""
tags = ["CNN","DeepLearning"]
categories = []
math = "true"
+++

In this post, we will be covering. 



Kernel visualization : we see what kernel extracts, the features
we are using 3x3 because nvidia is working on it
11/9/7 we need to determine what receptive field we need, to extract edges or gradients 
how to decide number of kernels- expressiveness required, (variation is less then our model needs less kernels, but as there is more classes and more variation we need our model to be expressiveness. expressiveness is the quality)- inter and intra class variation ( eg : we need to identity dogs and which dog it is)- hardware capacity : if not capable to run in ram or gpu then we might not be able to run large kernels
we use strides in resource constraint
in an architecture, we have titan x and jitsannano- one way is 224x224x32|3x3x32x64(p=1)|224x224x64|MP|112x112x64- second way is 224x224x32|3x3x32x64(p=0,s=2)|112x112x64
in first way we need more computation and need more memorysecond way can be used in embedded hardware, we might loose some quality but we have a hardware constraint on it.
we use strides when we have some hardware constraint, otherwise never think of it until it is forced to you- it causes checkerboard issue- we are colvolving less pixels than convolving with stride 1, which means we are skipping some information while convolving- it causes a diffused image, blurry
benifits of 1x1 convolution- less computatiom requirements to reducing number of channels
we are giving weights to each 1x1 channel, after getting some receptive field of 11/9/7, so 1x1 is just single operation on input channel
gpu workingsimd : single instruction multiple data ( work parallelly)cpu workingsisd : single instruction single data ( work linear )
we try to fill our gpu ram completely, and decide what batch size we should keep.
what 3x3 is doing is extracting another information from the input image when we have already arrived at 512 channels, we dont need to go from 512 to 32 again, but if we use 1x1 we will merge the features (it's a grouper), we have already extracted some features in 512 channels, now we will use those channels to create mixture of features, which can later be apllied 3x3 again
reduces the burden of 3x3 kernels
let's say we have arrived at parts of objects and want to add a block that we reach to full complete object
there might be some background information in the images, that we might not need to identify the objects, so there might be some channels in previous layer that might be storing some background information. there might be a channel which is working for car channel and might not be good for say detecting elephant. 
what do we need from 3x3 kernel- extract information- filter out the channels also
what if- we throw away some channels which are not useful by using 1x1 channel- throw away mean, giving weights 0 to those vectors
we can use 1x1 to increase the number of channels but we need a purposesome of the arhitecturesuppose we need to reach from 256 to 512 channels
- 3x3x256x512- 1x1x256x32 | 3x3x32x32 | 1x1x32x512
this is being used by resnet
both 1x1 and MP are filtering out the features from the previus layer.when we are done with extracing features in a block we would like to reduce the number of channels and the shape of object as we have already extracted the features and want to work on those features only
trend is using MP first than 1x1, check the computation
a kernel might not be able to make decision about what to do with the data, so we apply some sort of activation function, which will decide what kind of data needs to pass on to next layers
we need non-linearity in the NNallows to act like a universal approximation function
sigmoid function causes vanishing gradient problemsigmoid 0<output<1
if a gradient in middle of layer becomes 0, that gradient will be passed on to its previous layer which is chain rule, all the previous layer will become 0
### working of backpropagation , effects of activation function on it
now, relu is defined as 0 when x is less than equal to zero, and x when x>0
we have computed relu function such that it is differentiable function for the use casein terms of mathematics, derivative of 0 wrt x=0 is not defined, but in deep learning while training the network we have derivative at 0 is 0.
MP is a kernel
what is a kernel : we have an input X we apply a function f on x to get output yy = f(x)
3x3 is kernel which will be applied to the function to get ysimilarly if we apply 2x2 max pooling process we are applying MP function to x to get yby that defination, we are defining MP as kernel.
opencv takes RGB channel and give us BGR channels. (KINDLY KEEP THIS IN MIND)
# Architectural Basics
fully connected layerwe don't use it in CNN
to use it we need to- flatten out the image to 1 d array

after flattening out the image, we don't know what the pixels are telling us, as the spatial information is lost.as we have different variation of an object, converting them into 1d array will always result different pixels values in the input layer for the same object.
so, if we so many different variations in an image, there is a possibility that we can get most of pixels activated.
these layers are then connected to next layer.suppose we have 3x3 image, we flatten out to 9 pixels and we want our output as 2 valuesto do so we need (9x2) matrix of weights
issue, very large number of parameters


we dont use FCNN(fully connected NN) in CNN(convolution NN)
each line connected is called as weight. in a network lines matter
The modern architecture can be looked as expand and squeeze network
inception : googleresent : microsoft
the total number increases from 64>128>256>512 
we can use FC layer on 1D array... and not on 2d array
## Softmax layer
normalized exponential function
check why softmax is not a probability function
it increases a confidence of a class
if you are in medical domain.... softmax can kill us.
softmax is no probability, it's likelihood
Loss function
negative log likelihood
where to use maxpoolingreduce to channelnot used close to each otheruse as far off from final layer
## batchnormalization## dropoutused after every layerkind of regularization
## learning ratewhy is there fluctuation in the graphyrt
