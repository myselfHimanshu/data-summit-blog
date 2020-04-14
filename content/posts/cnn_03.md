+++
draft = false
date = 2020-04-05T22:22:22+05:30
title = "Convolutional Neural Network Part 3 : Decision Making"
slug = ""
tags = ["CNN","DeepLearning"]
categories = []
math = "true"
+++

In our last post, we left our network's faith in Ant-Man. We will be covering how Ant-Man gonna come to save our network's life. We'll try to understand some issues related to strides. We'll cover how network makes decisions and in the end as usual getting our hands dirty with Pytorch.

Kindly go through these post before moving forward, as we are building up our intuations from scratch.

- [CNN Part 1 : Basic Concepts](https://myselfhimanshu.github.io/posts/cnn_01/)
- [CNN Part 2 : Neural Architecture](https://myselfhimanshu.github.io/posts/cnn_02/)

Ready ? Let's jump in.

<p align="center">
  <img src="https://media.giphy.com/media/qtXHW324gTKqQ/giphy.gif"/>
</p>

**What are my kernels extracting ?**

Whatever CNN visualization we see on the internet, are of the features that are extracted from kernels and not the kernels itself.

Because visualization of kernels makes no sense, we wouldn't be able to identify what actually a kernel might be doing. We are interested in what a kernel might be extracting though. Look below at 3x3 kernels visualization

<p align="center">
  <img src="https://github.com/myselfHimanshu/data-summit-blog/raw/master/images/cnn_blog_03/kernelvis.png"/>
</p>

We cannot find any pattern here. In the initial layers - Kernels extract the low level features i.e Edges , Gradients , Color etc. This does not mean something meaningful is extracted out of first layer itself. 3x3 is a very small area to actually gather any Perceivable information. Even at 5x5 (for large image sizes) we can't make our a lot of stuff.

Something meaningful to interpret comes when we we arrive at a proper receptive field to find something useful/meaning depending on how complex our dataset is. If the dataset is too simple - first one or two layers will itself give a meaningful context . If data is complex, we may need more layers.

In other words, when we convolve a 3x3 kernel on an image in the first layer, the receptive field is 3x3. The receptive field is so small that we wouldn't be able to identify what are these kernels extracting.

To see what kind of features our kernels are learning, we need some higher receptive field. By higher I mean like 11x11 or more ( we have arrived to this number by experimention). Generally at 11x11 receptive field, we will start seeing some features, gradients, edges, or patterns.

**Why visualizing these features are important for us ?**

Look at below image

<p align="center">
  <img src="https://github.com/myselfHimanshu/data-summit-blog/raw/master/images/cnn_blog_03/vis_layer1.png"/>
</p>

*a* and *c* show the visualization of the first and second layers from AlexNet. We find that :

- there are dead neurons without any specific patterns in first layer, indicated by red box, which leads us an insight that there many be high learning rate or not good weight initialization while training the network.
- second layer visualization shows aliasing artifacts. This could be caused by the large strides used in first layer convolution.

These visualization helps us in optimizing CNN architecture. The improvements can be then seen in *b* and *d*. Patterns in *b* are more distintive and patterns in *d* have no aliasing artifacts.

Please visit these links for more visualization: 

- [Distil : Feature Visualization](https://distill.pub/2017/feature-visualization/)
- [CS231n : Visualizing and Understanding](http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture12.pdf)

**What is Checkboard Issue?**

When using high strides, one unplesant behaviour we observe is so-called checkerboard issue.

<p align="center">
  <img width="50%" height="50%" src="https://github.com/myselfHimanshu/data-summit-blog/raw/master/images/cnn_blog_03/checkerboard.png"/>
</p>

This result from "uneven overlap" of convolution. 

We learned in the last post that when we convolve with the standard way (stride of 1), we cover most of the pixels 9 times.

<p align="center">
  <img src="https://miro.medium.com/max/790/1*NrsBkY8ujrGlq83f8FR2wQ.gif"/>
</p>

But, when we convolve with a stride of more than 1, we would be covering some pixels more than once. And this is not good, as we are creating an island of extra information spread around in a repeating pattern. The image below would help:

<p align="center">
  <img src="https://github.com/myselfHimanshu/data-summit-blog/raw/master/images/cnn_blog_03/checkboard2.png"/>
</p>

As you can see, we are spreading the information unevenly. We will cover this checkerboard issue again in super-Resolution algorithms.

Let's come back to the rescue part of our network. Do you remember what the problem was ? Let's recap.

### The parameters bombarding issue 

ThIS was our network architecture, I have updated it a little bit: 

|Object Size|Kernel Size|Output Size|Parameters|RF|
|----|----|----|----|----|
|400x400x3|(3x3x3)x32|398x398x32|864|3x3|
|398x398x32|(3x3x32)x64|396x396x64|18432|5x5|
|396x396x64|(3x3x64)x128|394x394x128|73728|7X7|
|394x394x128|(3x3x128)x256|392x392x256|294912|9X9|
|392x392x256|(3x3x256)x512|390x390x512|1179648|11X11|
|MP|||
|195x195x512|(?x?x512)x32|?x?x32|???|22x22|
|?x?x32|(?x?x32)x64|?x?x64|???|24x24
|...|...|...|...|


**How should we reduce the number of kernels?**

We cannot just add 32, 3x3 kernels as that would re-analyze all the 512 channels and give us 32 kernels.
We have 512 features now, instead of evaluating these 512 kernels and coming out with 32 new ones, it makes sense to combine them to form 32 mixtures. That is where 1x1 convolution helps us.

- It would be better to merge our 512 kernels into 32 richer kernels which could extract multiple features which come together.
- 3x3 is an expensive kernel, and we should be able to figure out something lighter, less computationally expensive method. 
- Since we are merging 512 kernels into 32 complex one, it would be great if we do not pick those features which are not required by the network to predict our images (like backgrounds).

Our Ant-Man : 1x1 convolution filter

<p align="center">
  <img src="https://github.com/myselfHimanshu/data-summit-blog/raw/master/images/cnn_blog_03/1x1conv.png"/>
</p>

Initially, 1 x 1 convolutions were proposed in the Network-in-network [paper](https://arxiv.org/abs/1312.4400). They were then highly used in the Google Inception [paper](https://arxiv.org/abs/1409.4842). A few advantages of 1 x 1 convolutions are:

- Dimensionality reduction for efficient computations.
- Efficient low dimensional embedding, or feature pooling.
- Applying nonlinearity again after convolution.
- 1x1 is merging the pre-existing feature extractors, creating new ones, keeping in mind that those features are found together.
- 1x1 is performing a weighted sum of the channels, so it can so happen that it decides not to pick a particular feature that defines the background and not a part of the object. This is helpful as this acts like filtering.

Let's look at our new network:

|Object Size|Kernel Size|Output Size|Parameters|RF|
|----|----|----|----|----|
|400x400x3|(3x3x3)x32|398x398x32|864|3x3|
|398x398x32|(3x3x32)x64|396x396x64|18432|5x5|
|396x396x64|(3x3x64)x128|394x394x128|73728|7X7|
|394x394x128|(3x3x128)x256|392x392x256|294912|9X9|
|392x392x256|(3x3x256)x512|390x390x512|1179648|11X11|
|MP|||
|195x195x512|(1x1x512)x32|195x195x32|16384|22x22|
|195x195x32|(3x3x32)x64|193x193x64|18432|24x24|
|193x193x32|(3x3x64)x128|191x191x128|73728|26x26|
|...|...|...|...|

Notice, we can form convolution blocks, receiving 32 channels and then perform 4 convolutions, giving finally 512 channels, which can then be fed to transition block (hoping to receive 512 channels) which finally reduces channels to 32. Also in this case, we have decreased the number of parameters as compared to the use of 3x3 kernels. 

### The Architectures

Most of the modern architecture follows this architecture:

**AlexNet**

<p align="center">
  <img src="https://github.com/myselfHimanshu/data-summit-blog/raw/master/images/cnn_blog_03/alexnet.png"/>
</p>

**VggNet**

<p align="center">
  <img src="https://github.com/myselfHimanshu/data-summit-blog/raw/master/images/cnn_blog_03/vggnet.png"/>
</p>

**ResNet**

<p align="center">
  <img src="https://github.com/myselfHimanshu/data-summit-blog/raw/master/images/cnn_blog_03/resnet.png"/>
</p>

ResNet is the latest among the above. Only the most advanced networks have ditched the FC layer for the GAP layer.

### Activation Function

When we convolve a kernel over the image, we get some output matrix. Now, the question arries is what do we do with those numbers. We need our network to make some decisions whether some pixels need to be forwarded into next layer or not.

This is where activation function comes in place. 

**Why do we need an activation function?**

If we do not apply an Activation function then 

- the output signal would simply be a simple linear function. 
- a linear function is just a polynomial of one degree. 
- a linear equation is easy to solve but they are limited in their complexity and have less power to learn complex functional mappings from data. 

We want our Neural Network to not just learn and compute a linear function but something more complicated than that.

Without activation function our Neural network would not be able to learn and model other complicated kinds of data such as images, videos , audio , speech etc.

<p align="center">
  <img src="https://github.com/myselfHimanshu/data-summit-blog/raw/master/images/cnn_blog_03/activationfunction.png"/>
</p>

The activation function is a mathematical gate in between the input neuron and output going to next layer. It can be simple as step function or it can be a transformation that maps the input neurons into output neurons that are needed for the network to learn.


**Why do we need non-linearity?**

We need a Neural Network Model to learn and represent almost anything and any arbitrary complex function that maps inputs to outputs. 

> Neural-Networks are considered Universal Function Approximators. 

It means that they can compute and learn any function at all. Almost any process we can think of can be represented as a functional computation in Neural Networks.

We need to apply an Activation function f(x) so as to make the network more powerful and add the ability to it to learn something complex and complicated form data and represent non-linear complex arbitrary functional mappings between inputs and outputs. Hence using a non-linear Activation we are able to generate non-linear mappings from inputs to outputs.

Also another important feature of a Activation function is that it should be differentiable. We need it to be this way so as to perform backpropogation optimization strategy while propogating backwards in the network to compute gradients of error(loss) with respect to weights and then accordingly optimize weights using Gradient descent or any other optimization technique to reduce error.

Activation functions also have a major effect on the neural network’s ability to converge and the convergence speed, or in some cases, activation functions might prevent neural networks from converging in the first place.

#### Types of Activation functions

Modern Neural Networks uses non-linear activation function, and that's what we will be focusing on.

**Sigmoid Activation function**

The sigmoid function is defined as,

|function|graph|
|----|----|
|$$\sigma(x) = \frac{1}{1+e^{-x}}$$|<p align="center"><img width="50%" height="50%" src="https://upload.wikimedia.org/wikipedia/commons/thumb/8/88/Logistic-curve.svg/1200px-Logistic-curve.svg.png"/></p>|

Advantages : 

- output values are bound between 0 and 1, normalizing the output of each neuron.
- clear predictions : for x above 2 or below -2, tends to bring the prediction value close to 0 or 1.
- function is differentiable. That means, we can find the slope of the sigmoid curve at any two points.

Disadvantages :

- Vanishing Gradient : for very high or very low values of X, there is no change in prediction.
- not zero centered, it makes gradient updates go too far in different directions. $0<output<1$ makes optimization harder.
- Computationally expensive.

**Tanh/Hyperbolic Tangent function**

The function is defined as,

|function|graph|
|----|----|
|$$ tanh(x) = \frac{2}{1 + e^{-2x}} - 1 $$ <br> OR <br> $$ tanh(x) = 2 * sigmoid(2x) - 1 $$|<p align="center"><img src="https://github.com/myselfHimanshu/data-summit-blog/raw/master/images/cnn_blog_03/tanh.png"/></p>|

Advantages :

- output is zero centered
- optimization is easier

Disadvantages :

- Same as sigmoid function

**ReLU (Rectified Linear Unit) activation function**

The function is defined as,

|function|graph|
|----|----|
|$$ f(x) = max(0,x) $$|<p align="center"><img src="https://github.com/myselfHimanshu/data-summit-blog/raw/master/images/cnn_blog_03/relu.png"/></p>|

Advantages :

- It’s cheap to compute as there is no complicated math. The model can, therefore, take less time to train or run.
- It converges faster. Linearity means that the slope doesn’t saturate, when x gets large. It doesn’t have the vanishing gradient problem suffered by other activation functions like sigmoid or tanh.
- It’s sparsely activated. Since ReLU is zero for all negative inputs, it’s likely for any given unit to not activate at all.

Disadvantages :

- The Dying ReLU problem—when inputs approach zero, or are negative, the gradient of the function becomes zero, the network cannot perform backpropagation and cannot learn.

**Leaky ReLU activation function**

The function is defined as,

|function|graph|
|----|----|
|$f(x) = ax $ for x<0 <br>AND<br> $f(x) = x$ for x>0|<p align="center"><img width="70%" src="https://github.com/myselfHimanshu/data-summit-blog/raw/master/images/cnn_blog_03/leaky-relu.png"/></p>|

Advantages :

- Prevents dying ReLU problem—this variation of ReLU has a small positive slope in the negative area, so it does enable backpropagation, even for negative input values
- Otherwise like ReLU

Disadvantages :

- Results not consistent—leaky ReLU does not provide consistent predictions for negative input values.

And there are other variations to relu.

#### Which Activation to use? 

<p align="center">
  <img width="20%" height="20%" src="https://media.giphy.com/media/lKXEBR8m1jWso/giphy.gif"/>
</p>

<p align="center">
  <img src="https://github.com/myselfHimanshu/data-summit-blog/raw/master/images/cnn_blog_03/val_activation.png"/>
</p>

ReLU is simple, efficient and fast, and even if anyone of the above is better, we are talking about a marginal benefit, with an increase in computation.

#### Pytorch 101

For getting hands dirty, kindly follow this link : [Pytorch 101](https://colab.research.google.com/github/dvgodoy/PyTorch101_ODSC_London2019/blob/master/PyTorch101_Colab.ipynb)

Stay tuned for the next post, Happy Learning!!!

If you feel that I can provide you with value, I encourage you to connect with me, follow me, add me to your circles etc.