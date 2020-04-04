+++
draft = false
date = 2020-03-22T22:22:22+05:30
title = "Convolutional Neural Network Part 1 : Basis Concepts"
slug = ""
tags = ["CNN","DeepLearning"]
categories = []
math = "true"
+++

I am starting this series of posts, where in we'll build on background knowledge of neural networks and explore what CNNs are. We will cover from basic understanding of how and what is CNN, augumentation techniques, architectures of CNNs, training an object detection model and more. Before moving forward, I recommend you to learn some basic terminologies of neural networks and how they work.

> If you're offered a seat on a rocket ship, don't ask what seat! Just get on.

Ready ? Let's jump in.

<p align="center">
  <img src="https://media.giphy.com/media/kgFZ6dOUGnYBO/giphy.gif">
</p>


Convolutional Neural Networks are very similar to ordinary Neural Networks. 

1. A neuron receives some inputs 
2. performs a dot product
3. follows it with a non-linearity 

The complete network expresses a single differentiable score function.

**So why not use a normal Neural Network?**

Images used in these kind of problems are often 224x224 or larger sizes. Imagine building a neural network to process 224x224 color images. That will be

$$ 224 * 224 * 3 = 150528 $$ 

input features. 3 here are color channels (RGB) in an image.

Let's say you have 1024 neurons in next hidden layer, then we might have to train 

$$ 150528 * 1024 \approx 154M $$

150+ million parameters for the first layer only. 

<p align="center">
  <img src="https://media.giphy.com/media/keZhECYHtGty4Jh1Vo/giphy.gif">
</p>

The next reason is that positions can change. You want your network to detect a dog in an image irrespective of where it is. What I mean is, a dog can be in a corner of image or small dog or whether it is a close-up shot.
These kind of images, would not activate the same neurons in the network, so network would react differently.   

The important thing about images are that <b> pixels are most useful features in the context of their neighbors </b>.

ConvNet architectures make the explicit assumption that the inputs are images. These then make the forward function more efficient to implement and vastly reduce the amount of parameters in the network.


<b> Your brain might be fooling you !!! </b>

Here, check out this image.

<p align="center">
  <img src="https://cdn.fstoppers.com/styles/large-16-9/s3/lead/2019/08/eb7ff9947f527a2e84d7f06e79138b82.png">
</p>

The above image is a black and white image, with color grids on it. Zoom out and in to see the magic. 

> Color is not a dependable feature in CNN.

## Terminologies

Every image can be represented as a matrix of pixel values.

<p align="center">
  <img width="250" height="250" src="https://www.apsl.net/media/apslweb/images/gif-8.original.gif">
</p>

### What are Channels ?

Channel is a convolutional term used to refer to certain feature of an image. In practicality, an image from standard digital camera will have 3 channels (RGB (red, green, blue)). An image printed on a newspaper has 4 channels (CMYK). You can imagine three 2d matrices over each other, each having pixels values in range of [0,255].

<p align="center">
  <img src="https://static.packt-cdn.com/products/9781789613964/graphics/e91171a3-f7ea-411e-a3e1-6d3892b8e1e5.png">
</p>

A grayscale image, has just one channel. The value of each pixel in the matrix ranges from 0 to 255 – zero indicating black and 255 indicating white.

| ![](https://upload.wikimedia.org/wikipedia/en/4/4c/Channel_digital_image_RGB_color.jpg) | ![](https://upload.wikimedia.org/wikipedia/en/4/45/Channel_digital_image_red.jpg) | <img src="https://upload.wikimedia.org/wikipedia/en/a/a8/Channel_digital_image_green.jpg" /> | ![](https://upload.wikimedia.org/wikipedia/en/b/b0/Channel_digital_image_blue.jpg) |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| RGB image                                                    | Red channel <br />(converted into grayscale)                 | Green channel<br />(converted into grayscale)                | Blue channel<br />(converted into grayscale)                 |

In above image, the red dress is much brighter in the red channel than in the other two, and the green part of the picture is shown much brighter in the green channel.

You shouldn't worry about RGB or CMYK specific channels. They are just like metrices which define an image. The main concept here is `channel`. An image can be divided into any number of channels, for an example, slide projector. You can imagine each slide as a channel and overlapping of these slides will form an image that gets projected over a screen.

Working on a specific color feature may be useful when you are asking a question to the network like, find me all yellow color flowers.

> What is a channel? <br>
> A channel is set of relatable features.

><b>Channel</b> Synonym :<br>
> <i>feature map</i>, <i>convolved feature</i>, <i>activation map</i>

Let's take this image as an example to explain what a channel can be.

<p align="center">
  <img src="https://d2v9y0dukr6mq2.cloudfront.net/video/thumbnail/2QQOex4/deep-learning-animated-word-cloud_s8oppv-il_thumbnail-full07.png">
</p>

Now imagine that we have 26 channels for the given image. Let's say our channels are A-Z, as there are 26 alphabets so 26 channels.

- Channel A : where in all the a from image are filtered out (wherever and however they are, just a )
- Channel B : where in all the b from image are filtered out.
- and so on, till Channel Z.


Let's talk about Channel A. Now a particular <i>a</i> in that channel is called `feature` , it can be big, small, tilted, anything but same feature. 

Now, when I asked you to filter out just <i>a</i> or extract just a single alphabet from the image to create a channel, you might need an extractor to do so. This extractor is termed as `kernel` . If you need to extract say <i>c</i>, you need this <i>c</i> filter. 

> <b>Kernel</b> Synonym : <br>
> <i>feature extractor</i>,
> <i>n x n matrix</i>,
><i>filter</i>,
> <i>weights</i> 

> Each filter gives us a channel. <br>

### Building blocks of Convolutional Neural Network

The primary purpose of Convolution is to extract features from the input image. As we have learned, how to speak a word by first learning what is an alphabet, similarly there are building blocks in CNN.

Like in english, we can define building blocks as : 

> alphabets combined => words combined => sentences combined => paragraphs combined => stories

similarly in vision,

> gradient and edges combined => textures and patterns combined => part of objects combined => complete objects => scenes

Look at below image, how does a network learn to identify each block that is mentioned above step by step.

<p align="center">
  <img src="https://github.com/myselfHimanshu/data-summit-blog/raw/master/images/cnn_01_01.png">
</p>

### What are gradients and edges ? 

We refer `gradient` as change in brightness over a series of pixels. 

<p align="center">
  <img width="500" height="180" src="https://huaxin.cl/wp-content/uploads/2018/09/gradient-black-to-white-wallpaper-4.jpg">
</p>

The above image is linear gradient from black to white. This gradient is smooth and we wouldn't say there is edge in this image. Edges are a bit fuzzy in above image. In real objects those `edges` can be found by looking at sharp transitions. You can find edges in the image below

<p align="center">
  <img width="500" height="180" src="https://upload.wikimedia.org/wikipedia/commons/thumb/2/20/%C3%84%C3%A4retuvastuse_n%C3%A4ide.png/500px-%C3%84%C3%A4retuvastuse_n%C3%A4ide.png">
</p>

### What is Convolution ?

Convolution takes care of spatial relationship between pixels by learning features from input image.
Let's see, what human eyes sees and what machine sees when looking at an image.

<p align="center">
  <img src="https://ai.stanford.edu/~syyeung/cvweb/Pictures1/imagematrix.png">
</p>

- First image is what you see; 
- Second image is how an image is interpreted (matrix of pixels values); 
- Third image is what machine sees ( n x n matrix of numbers).

**How does convolution preserves spatial relationship ?**

Consider, a 5x5 image whose pixel values are only 0 an 1.

<p align="center">
  <img src="https://github.com/myselfHimanshu/data-summit-blog/raw/master/images/cnn_blog_01/image1.png">
</p>

and also conside 3 x3 matrix as shown below

<p align="center">
  <img src="https://github.com/myselfHimanshu/data-summit-blog/raw/master/images/cnn_blog_01/image2.png">
</p>

Then, convolution of 5x5 image and 3x3 image can be computed as shown below : 

<p align="center">
  <img src="https://icecreamlabs.com/wp-content/uploads/2018/08/33-con.gif">
</p>

Let's understand what is happening ? How the computation is being processed ?

We slide the yellow matrix over our green image by 1 pixel, and for every position we compute element wise multiplication (between two matrices) and add up the multiplication outputs to get a final integer which forms an element in our pink convolved feature matrix.

There are some terminologies for each step that is happening above.

Suppose we have blue box as our image and the dark patch as our yellow matrix and white box as our convolved feature matrix.

<p align="center">
  <img src="https://cdn-images-1.medium.com/max/1024/1*Fw-ehcNBR9byHtho-Rxbtw.gif">
</p>

The blue object is your image of size (5x5) ( 5 is your width and height) , the dark patch that you see is our `kernel`, which is initialized with random values.

There is dot product between 3x3 filter and 3x3 patch of image. This filter slides over the image 1 step at a time. 

This <i>step</i> is termed as `stride` (here stride=1) and the results of dot product is creating our output channel. 


>This process of n x n filter rolling over on top of the image and computing dot product step by step is termed as <b>convolution</b>.


These kernels are initialised with random values in the beginning. These values will change over in model training. By model I mean is, when you train your deep learning model. Different values of the filter matrix will produce different feature maps for same input image.

Consider another image,

<p align="center">
  <img src="https://github.com/myselfHimanshu/data-summit-blog/raw/master/images/cnn_blog_01/image11.png">
</p>

Let's see effect of different filters over above image,

| Operation | Filter | Convolved Image|
| :-----: | :-----: | :-----: |
|Identity|$$\begin{pmatrix}0 & 0 & 0\\\ 0 & 1 & 0\\\0 & 0 & 0\end{pmatrix}$$|![](https://github.com/myselfHimanshu/data-summit-blog/raw/master/images/cnn_blog_01/image11.png)|
|Vertical Edge|$$\begin{pmatrix}-1 & 0 & 1\\\ -1 & 0 & 1\\\ -1 & 0 & 1\end{pmatrix}$$|![](https://github.com/myselfHimanshu/data-summit-blog/raw/master/images/cnn_blog_01/image7.png)|
|Horizontal Edge|$$\begin{pmatrix}-1 & -1 & -1\\\ 0 & 0 & 0\\\1 & 1 & 1\end{pmatrix}$$|![](https://github.com/myselfHimanshu/data-summit-blog/raw/master/images/cnn_blog_01/image8.png)|
|Horizontal Edge Sharpen|$$\begin{pmatrix}-4 & -4 & -4\\\ 0 & 0 & 0\\\ 4 & 4 & 4\end{pmatrix}$$|![](https://github.com/myselfHimanshu/data-summit-blog/raw/master/images/cnn_blog_01/image9.png)|
|Blured Edge|$$ \frac19 \begin{pmatrix}1 & 1 & 1\\\ 1 & 1 & 1\\\1 & 1 & 1\end{pmatrix}$$|![](https://github.com/myselfHimanshu/data-summit-blog/raw/master/images/cnn_blog_01/image10.png)|

In practice, a CNN learns the values of these filters during training process.

**See what's happening ?**

Vertical edge filter detects vertical edges and horizontal edge filter detects horizontal edges. A bright pixel in output image indicates that there is strong edge around there in original image.

### Output Channel

When we convolve 3x3 filter on 6x6 image we get 4x4 object matrix. 

Whatever we are getting as a result, we are applying other filter over it to get next set of objects. (I am talking about layers of Neural Networks).

The size of the output channel can be calculated using following formula,

$$
    n_o = \lfloor\frac{n_i+2p-k}{s}\rfloor + 1
$$

$n_o$ : number of output features in a dimension<br>
$n_i$ : number of input features in a dimension<br>
$k$   : kernel size<br>
$p$   : padding size<br>
$s$   : stride size

I will explain what is padding and why we use it later.

### Parameters of kernels

These kernels have values which are termed as <i>parameters</i>. We can say a 3x3 kernel will have 9 parameters.

Now coming to some mathematical concepts. 

- If we use a 5x5 kernel for convolution on 5x5 image, we will get output object of size 1x1. 
- If we use a 3x3 kernel for convolution on 5x5 image, we will get output object of size 3x3 and again convolving the output with 3x3 kernel will give us output object of size 1x1. 

So I can say, convolving with 5x5 kernel is same as convolving with 3x3 kernel (twice).

Now the question arises is, **What size kernel to use ?**

The answer to above question lies in another question, 

**How many parameters we have to train on if we use 3x3 or 5x5 kernel?**

 Look at the table below which will tell you about total number of parameters to train if we use n x n size kernel.

| N <br />(N x N image) | total number of Parameters <br />( 3x3 kernel ) | total number of Parameters <br />( N x N kernel ) |
| :---------------------: | :-----------------------------------------------: | :-------------------------------------------------: |
|            5            |                   (3x3)*(2) = 18                   |                     (5x5) = 25                      |
|            7            |                   (3x3)*(3) = 27                   |                     (7x7) = 49                      |
|            9            |                   (3x3)*(4) = 36                   |                     (9x9) = 81                      |
|           11            |                   (3x3)*(5) = 45                   |                    (11x11) = 121                    |

The above table tells us why using 3x3 kernel is better than the other size kernels.

Less parameters means, model will take less time to train on. It will be faster. We need to build an CNN architecture with less parameters and that gives us best result.

There will be tradeoffs between parameters and result. But we need to come up with an elegant architecture and accordingly we will select our kernel size.

In many other blogs or research papers, you might see researchers using odd shaped kernel.

**Is it necessary to just use odd number shaped kernel?** 

It make sense to have our kernel odd shaped because it has axies of symmetry. 

Let's draw a black line on a piece of paper. If I want my machine to learn the difference between white line and black line. I need to tell it that left side of line is white and right side is also white and the middle column is black. The machine needs to differentiate the black pixels with what is not line i.e white pixels. We need to provide both the information. The machine needs to know the start and the end of a feature.

You can check above table of edge detection, what values our kernels are made up of? and what is the output.

### Receptive Field

This is one of most important concept in CNNs. 

> A receptive field is defined as region in the input that a particular CNNs feature is looking at.

<p align="center">
  <img src="https://www.researchgate.net/publication/316950618/figure/fig4/AS:495826810007552@1495225731123/The-receptive-field-of-each-convolution-layer-with-a-3-3-kernel-The-green-area-marks.png">
</p>

Let's say Layer1 is your image of size 5x5 and green color matrix is your kernel of size 3x3. 

If you convolve the kernel on image, you will get 3x3 object which is your Layer2 and convolving again on this Layer2 you will get object of size 1x1, which is your Layer3.

Let's take green cell from Layer2. 

How many cells can it see in Layer1? The answer is 9. 

So, the `local receptive field` of that cell is 3. 

- If I ask that cell what is in there in (25,25) cell of Layer1 ? 
    - Answer : "I have no clue". 
- Now If I ask the yellow cell from Layer3, how many cells it can see ? 
    - Answer : "I have seen the whole image". 

**Wondering how ?**

A 5x5 kernel might have seen the whole image and as explained earlier, using 3x3 kernel (twice) is similar to using 5x5 kernel. So, the `global receptive field` of that yellow cell from Layer3 will be full 5x5 image and the local receptive field of a cell will always be the size of kernel. 

So the last cell should have seen the whole image, otherwise it wouldn't know if cat is in image or not.

Now, if we have to build a network for 401x401 image, 

<i>How many layers would we need by just using 3x3 kernel</i>? 

| image size | kernel size | output size | Global <br />Receptive field size |
| :--------: | :---------: | :---------: | :-------------------------------: |
| 401 x 401  |    3 x 3    |  399 x 399  |               3 x 3               |
| 399 x 399  |    3 x 3    |  397 x 397  |               5 x 5               |
| 397 x 397  |    3 x 3    |  395 x 395  |               7 x 7               |
|    ...     |    3 x 3    |     ...     |                ...                |
|   3 x 3    |    3 x 3    |    1 x 1    |             401 x 401             |

The answer will be, 200 layers.

This is not a nice way to train a model. Adding 200 layers is a nightmare. So, we have to downsample in between the layers also termed as `max-pooling` . We will go through this concept in next post.

### Initializing kernel values

The values of a neural network must be initializes to random numbers.

Understand these 2 concepts :

<b>Deterministic Algorithms vs Non-Deterministic Algorithms</b>

Given an unordered array of numbers, a bubble sort algorithm will always execute in same way to give same ordered result.  

But some problems cannot be solved by this technique efficiently because of complexity of data. The algorithm may run but may never give you required solution or might run infinitely.

To solve such problems, we use non-deterministic algorithms.

<p align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/1/16/Difference_between_deterministic_and_Nondeterministic.svg/950px-Difference_between_deterministic_and_Nondeterministic.svg.png">
</p>

A deterministic algorithm that performs f(n) steps always finishes in f(n) steps and always returns the same result. A non deterministic algorithm that has f(n) levels might not return the same result on different runs. A non deterministic algorithm may never finish due to the potentially infinite size of the fixed height tree.

These non-deterministic algorithm will arive at approximate solution but will be fast. These solution will often be satisfactory for such problems.

These kind of algorithms make use of randomness. You might have studied about gradient descent algorithm, these are referred to as [stochastic algorithms](https://en.wikipedia.org/wiki/Stochastic_optimization).

The process of finding solution is incremental, starting from a point in sapce of possible solutions to good enough solution. As we know nothing about space, we start with random chosen point.

Neural Networks are trained using these kind of algorithms. 

> Training algorithms for deep learning models are usually iterative in nature and thus require the user to specify some initial point from which to begin the iterations. Moreover, training deep models is a sufficiently difficult task that most algorithms are strongly affected by the choice of initialization.<br>

> Perhaps the only property known with complete certainty is that the initial parameters need to break symmetry between diﬀerent units. If two hidden units with the same activation function are connected to the same inputs, then these units must have diﬀerent initial parameters. If they have the same initial parameters, then a deterministic learning algorithm applied to a deterministic constant model will constantly update both of these units in the same way. <br> - page 296,297, [deep learning book](https://www.deeplearningbook.org/contents/optimization.html).

A careful initialization of the network can speed up the learning process.

<hr>








