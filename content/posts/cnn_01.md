+++
draft = false
date = 2020-03-22T22:22:22+05:30
title = "Convolutional Neural Network Part 1 : Basis Concepts"
slug = ""
tags = ["CNN"]
categories = []
math = "true"
+++

## Convolutional neural network

<hr>

As I am starting this CNN series of posts. I would like you to know about how neural networks work and then continue reading.

Convolutional Neural Networks are very similar to ordinary Neural Networks. Each neuron receives some inputs, performs a dot product and optionally follows it with a non-linearity. The whole network still expresses a single differentiable score function: from the raw image pixels on one end to class scores at the other. And they still have a loss function (e.g. SVM/Softmax) on the last (fully-connected) layer and all the tips/tricks we developed for learning regular Neural Networks still apply.

So what changes? ConvNet architectures make the explicit assumption that the inputs are images, which allows us to encode certain properties into the architecture. These then make the forward function more efficient to implement and vastly reduce the amount of parameters in the network.

The above paragraph is taken directly from, source : https://bit.ly/3bBizE7. Go through entire CS231n later.

<hr>

I'll try to explain all the concepts in a very layman's term. If you feel like you didn't understand any term or concept in this following series, kindly comment at the end of post in the section. I'll try to refactor the concept to make it more understandable.

```text
Things to keep in mind :

1. Share things
2. Ask questions
3. Research 
4. Build and deploy model end to end.
5. Magic of deep learning is in it's loss function.
6. Mathematics is important. 
   Try to answer why are you using this function?
7. Machine Learning and Deep Learning are IT field.
```

### Your brain might be fooling you

To start with I would like to show you an image. An image that will confuse you. Your brain is fooling you for now maybe because you might have never seen an image like this.

![](https://cdn.fstoppers.com/styles/large-16-9/s3/lead/2019/08/eb7ff9947f527a2e84d7f06e79138b82.png)


The above image is a black and white image, with color grids on it. Zoom out and in to see the magic. 

From this we learn that `Color is not a dependable feature in CNN.`

### What are Channels ?

In practicality, most input images have 3 channels (RGB), 1 channel for grayscale images.

| ![](https://upload.wikimedia.org/wikipedia/en/4/4c/Channel_digital_image_RGB_color.jpg) | ![](https://upload.wikimedia.org/wikipedia/en/4/45/Channel_digital_image_red.jpg) | <img src="https://upload.wikimedia.org/wikipedia/en/a/a8/Channel_digital_image_green.jpg" /> | ![](https://upload.wikimedia.org/wikipedia/en/b/b0/Channel_digital_image_blue.jpg) |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| RGB image                                                    | Red channel <br />(converted into grayscale)                 | Green channel<br />(converted into grayscale)                | Blue channel<br />(converted into grayscale)                 |

In above images, the red dress is much brighter in the red channel than in the other two, and the green part of the picture is shown much brighter in the green channel.

There can be RGB, CMYK(printing an image on newspaper) types of channels. These RGB, CMYK you can think it is of some metrics i.e cm, km, mm. and as said earlier color is not a dependable feature in CNN.

You can divide an image in any number of channels eg. slide projector, where different slides can form an image over a screen, channels in this example can be the slides.

Colors may be useful when you are asking a question like `find me all yellow color flowers` . 

The key components to learn here are `gradients` and `edges`.

<b>
A `channel` is set of relatable features. <br>
A `channel` can also be termed as `feature map`.
</b>

I'll explain, what a feature is in below sections.

For an example, 

![](https://d2v9y0dukr6mq2.cloudfront.net/video/thumbnail/2QQOex4/deep-learning-animated-word-cloud_s8oppv-il_thumbnail-full07.png)

Let's divide this image into channels. Let's say our channels are A-Z.

Channel A : where in all the `a` s from image are filtered out (wherever they are and however they are, just `a`s )

Channel B : where in all the `b` s from image are filtered out.

and so on, till Channel Z.

Let's talk about Channel A. Now a particular `a` in that channel is called `feature` , it can be big, small, tilted, anything but same feature. 

Now, when I asked you to filter out just `a` or extract just a letter from the image, this extractor is termed as `filter` . If you need to extract say `c`, you need this `c filter` extractor. 

<b>
These filters can also be termed as: <br>
    - feature extractor<br>
    - n x n matrix<br>
    - kernel<br>
    - weights 
</b>

Each `filter` will give a `channel`. 

### Building blocks of CNN

Follow this in vision:

Like in English 

`alphabets =make> words =make> sentences =make> paragraphs =make> stories`

similarly in vision

`gradient and edges combined =make> textures and patterns combined =make> part of objects combined =make> complete objects =make> scenes`

Get a general overall view of the steps and the below image depicts how machine learn to identify images...

![](https://github.com/myselfHimanshu/data-summit-blog/raw/master/images/cnn_01_01.png)

source : https://distill.pub/2017/feature-visualization/

### What are gradients and edges ? 

We refer "gradient" as change in brightness over a series of pixels. 

<img src="https://lh3.googleusercontent.com/proxy/2PBEIWIjfceFVKJiOI883rs8b6G6UQ5wqh8VGtg2epSWrL_wXeZ6bO5ReKBfF1XhRzf7o-K2aOerVQVR12DfDWdQ1MfMzuoAnAtdJSSnS-yLDA" style="zoom:50%;" />

The above image is linear gradient from black to white. This gradient is smooth and we wouldn't say there is "edge" in this image. Edges are a bit fuzzy in above image. In real objects those edges can be found by looking at sharp transitions. You can find edges in the image below

![](https://upload.wikimedia.org/wikipedia/commons/thumb/2/20/%C3%84%C3%A4retuvastuse_n%C3%A4ide.png/500px-%C3%84%C3%A4retuvastuse_n%C3%A4ide.png)


### What is Convolution ?

First let's see, what human eyes sees and what machine sees when looking at an image.

![](https://ai.stanford.edu/~syyeung/cvweb/Pictures1/imagematrix.png)

First image is what human eye sees, second one is how an image is interpreted (set of numbers also pixels) and third image is what machine sees( n x n matrix of numbers).

The above is an image represented as a matrix of pixel values. Each pixel value ranges between [0,255], with 0 being black and 255 being white.

From now onwards, we will talk about these matrixes. 

We were talking about kernels and features, look at the image below

![](https://cdn-images-1.medium.com/max/1024/1*Fw-ehcNBR9byHtho-Rxbtw.gif)

The blue object is your image of size (5x5) ( 5 is your width and height) , the dark patch that you see is a `kernel` or `filter` or `3x3 matrix` , which is initialized with random values.

There is dot product between `3x3 filter` and `3x3 patch` of image that it is moving on step by step. This `step` is termed as `stride` (here `stride=1`).  The results of dot product is creating that `white` object or `white channel`. 


<b>This process of `n x n filter` rolling over on top of the image and computing dot product step by step is termed as `convolution`.</b>


These `kernels` are initialised with random values in the beginning. These values will change over in model training. By model I mean is, when you train your deep learning model.

Here is the visualization of convolution,

![](https://icecreamlabs.com/wp-content/uploads/2018/08/33-con.gif) 

or check another intuative example:

![edge_detection](https://github.com/myselfHimanshu/data-summit-blog/raw/master/images/edge_detection.png)


so if we are convolving `3x3 filter matrix` on `6x6 object matrix` we get `4x4 object matrix`. Whatever we are getting as a result, we are applying other filter over it to get next set of objects. (I am talking about layers of Neural Networks).

Why we are using these `kernels`? As explained above, a `kernel` is a `feature extractor`.

### Parameters

These `kernels` have values which are termed as `parameters`. So, a `3x3 kernel` has `9 parameters`.

If we use `5x5 kernel` on `5x5 image` , you would have got `1x1 object`. So I can say, convolving with `5x5 kernel` is same as convolving with `3x3 kernel (twice)`. 

Now the question arises is, how many parameters we have to train if we use `3x3` or `5x5` kernel. 

As, less parameters means, model will take less time to train. We need to build an CNN architecture with less parameters which will give us best result. 

 Look at the table below which will tell you about total number of parameters to train if we use `n x n size` kernel.

| N <br />`(N x N image)` | total number of Parameters <br />`( 3x3 kernel )` | total number of Parameters <br />`( N x N kernel )` |
| :---------------------: | :-----------------------------------------------: | :-------------------------------------------------: |
|            5            |                   (3x3)(2) = 18                   |                     (5x5) = 25                      |
|            7            |                   (3x3)(3) = 27                   |                     (7x7) = 49                      |
|            9            |                   (3x3)(4) = 36                   |                     (9x9) = 81                      |
|           11            |                   (3x3)(5) = 45                   |                    (11x11) = 121                    |

The above table tells us why using `3x3 kernel` is better than the other size kernels. 

Do we have to use odd number shaped kernel ? it make sense to have our kernel odd shaped because it has axies of symmetry. 

Let's draw a black line on a piece of paper. If I want my machine to learn the difference between white line and black line. I need to tell it that left side of line is white and right side is also white and the middle column is black. The machine needs to differentiate the black pixels with what is not line i.e white pixels. We need to provide both the information. The machine needs to know the start and the end of a feature.

### Receptive Field

![](https://www.researchgate.net/publication/316950618/figure/fig4/AS:495826810007552@1495225731123/The-receptive-field-of-each-convolution-layer-with-a-3-3-kernel-The-green-area-marks.png)

Let's say `Layer1` is your `image size 5x5` and green color matrix is your `kernel 3x3` . If you convolve on image, you will get `3x3 object` which is your `Layer2` and convolving again on this `Layer2` you will get `1x1 object` which is your `Layer3`.

Let's take green cell from Layer2. 

How many cells can it see in Layer1? The answer is 9. 

So, the `local receptive field` of that cell will be 9. 

If I ask that cell what is in there in (25,25) cell of Layer1 ? The answer it will give is `I have no clue`. 

Now If I ask the yellow cell from Layer3, how many cells it can see ? The answer will be `it has seen the whole image`. Wondering how ? 

A `5x5 kernel` might have seen the whole image. And as explained earlier, using `3x3 kernel (twice)` is similar to using `5x5 kernel`. So, the `local receptive field` of a cell will always be the size of `kernel` and  the `global receptive field` of that yellow cell from Layer3 will be full `5x5 image`. 

So the last cell should have seen the whole image, otherwise it wouldn't know if cat is in image or not.

Now, if we have to build a network for `401x401 image` , how many layers would we need using `3x3 kernel`? 

| image size | kernel size | output size | Global <br />Receptive field size |
| :--------: | :---------: | :---------: | :-------------------------------: |
| 401 x 401  |    3 x 3    |  399 x 399  |               3 x 3               |
| 399 x 399  |    3 x 3    |  397 x 397  |               5 x 5               |
| 397 x 397  |    3 x 3    |  395 x 395  |               7 x 7               |
|    ...     |    3 x 3    |     ...     |                ...                |
|   3 x 3    |    3 x 3    |    1 x 1    |             401 x 401             |

The answer will be, `200` layers.

This is not a nice way to train a model. Adding 200 layers is a nightmare. So, we have to downsample in between the layers also termed as `max-pooling` . We will go through every concept.

#### Some brainstorming questions.

#### Questions

1. What are Channels and Kernels?
2. Why should we (nearly) always use 3x3 kernels?
3. How many times to we need to perform 3x3 convolutions operations to reach close to 1x1 from 199x199 (type each layer output like 199x199 > 197x197...)
4. How are kernels initialized? Why are these `kernels` initialised randomly at the beginning and not any other specific intialization?
5. What happens during the training of a DNN?









