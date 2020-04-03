+++
draft = false
date = 2020-03-29T22:22:22+05:30
title = "Convolutional Neural Network Part 2 : Neural Architecture"
slug = ""
tags = ["CNN"]
categories = []
math = "true"
+++

In this tutorial, we will be training the network on MNIST dataset.  We'll try to understand neural network architecture. 

The most important thing to understand in deep learning is what goes in and what comes out, with the understanding of what the network is doing inside that black box. Here black box refers to the hidden layers of DNN. What is it seeing first and at the end of the training the model.

The below architecture is not meant to give us very good score. In this series we'll only focus on pytorch-101. To understand the basic terminologies of CNN building blocks. Kindly go through this post.

[CNN Part 1 : Basic Concepts](https://myselfhimanshu.github.io/posts/cnn_01/)

Ready ? Let's jump in.

<p align="center">
  <img src="https://media.giphy.com/media/11OWKkvYUmZQOs/giphy.gif"/>
</p>


## Neural Architecture : Part II

In the below section we are introducting two main concept. 

**Max Pooling** and **Receptive Field**

Before jumping further, let me tell you **what padding is?**

You can think padding as extra rows and columns of pixels that are applied around a feature map.

<p align="center">
  <img src="https://github.com/myselfHimanshu/data-summit-blog/raw/master/images/cnn_blog_02/padding.jpg">
</p>

In the above image, we have a feature map of 5x5, if we apply a padding=1, we add a row and a column around that feature map. This feature map will result in size of 7x7 now on which we apply convolution.

It is not necessary to apply padding with value 0, as shown in the image. We will see what padding should we apply such that we can gain more information from around the corners of the feature map.

Now, in the last post we have calculated how many layers we have to use if we use 3x3 kernel on a 401x401 image size with stride=1 and padding=0?

The answer was 200.

**Do we really need 200 layers in our network?**

The answer is No.

We need our model to learn fast and learn accurate, for which we should built an architect wherein our last layer's receptive field should be the whole object. What I mean is, we want our network's receptive field to slowly increase as we add layers. Before taking any decision, the whole image needs to be processed.

> Refer to Building blocks of Convolutional Neural Network section of [previous post](https://myselfhimanshu.github.io/posts/cnn_01/).

To reduce number of layers, one technique to downsample a feature map is `MaxPooling`.<br><br><br>


![](https://cdn-images-1.medium.com/freeze/max/1000/1*ghJyfuw-9a5esjJqGuBggA.jpeg?q=20)

It is clear what is happening from the above image.

If we look at first row, applying a max-pool layer of size 2x2 on 4x4 feature map with stride equal to the size of the layer gives 2x2 feature map.

**MaxPooling** is used for dimension reduction. It is a simple layer where no learning happens.

It helps in reducing the number of learned parameters, which helps in reducing the computation and memory load.

> We need to take care of where to apply max pooling layer in the network.

**Why is above line important?**

Let's work on an image of 4. Here we apply one max pool layer of size 2x2 over first image, which results second image and applying another max pool layer we get image 3.

|image|apply maxpool|apply maxpool again|
|-----|------|------|
|![](https://github.com/myselfHimanshu/data-summit-blog/raw/master/images/cnn_blog_02/maxpool_org.png)|![](https://github.com/myselfHimanshu/data-summit-blog/raw/master/images/cnn_blog_02/maxpool_2.png)|![](https://github.com/myselfHimanshu/data-summit-blog/raw/master/images/cnn_blog_02/maxpool3.png)|

The above process is applying maxpooling again and again on the same image. This doesn't happen in the network. There will convolution in middle. The concept I am explaining you is about where to use maxpool layer and what will happen if we start applying maxpooling without knowing what the network has learned in previous layer. 

Now ask what is your network learning if I do this?

It might be learning bananas for minions.

<p align="center">
  <img src="https://media.giphy.com/media/ZqlvCTNHpqrio/giphy.gif"/>
</p>

So we need to really careful, where should I apply the maxpool layer.

> Never use max pooling close to your output/prediction layer, the network might end up loosing important features.

**Are there any other effects on feature maps if we apply max pooling layer?**

I am glad you asked this. The answer is Yes.

Max pooling adds a bit of shift variance. What is shift invariance?

Suppose you have an image of a dog. And it is wagging it's tail. Now just imagine with me this,

there is a feature map of size 5x5, where in middle column number 3 values are 1 and everything else 0. That middle column is the dog's tail.

Now, it waggle the tail and the feature of tail is shifted to column 4. If we apply max pooling on both of these feature maps, we will get same result. This is shift invariance. The data has changed but it doesn't matter.

If the tail shifts in an image, do I tell that the dog is without tail? No!! right? In both of these cases we still need to identify that it was the tail.

Max pooling takes care of very small data invariance. Not talking about large invariance, those things are learned by other kernels.

This is just a concept, don't worry I'll explain in future posts if it arrives. Don't think about whether it is a good thing or bad thing for now.

There are other invariances like, rotational invariance, scale invariance. Right now, we are not going in depth of these terminologies.

Coming back to the question, **How many layers do I need now when I add maxpooling layer?**

Given:

We have image size of 400x400, kernel size of 3x3, stride = 1, padding = 0 and max pool layer (MP) of size 2x2 with stride 2.

Here, we will write down our output object size,

400 | 398 | 396 | 394 | 392 | 390 | MP (2x2) <br>
195 | 193 | 191 | 189 | 187 | 185 | MP (2x2) <br>
92 | 90 | 88 | 86 | 84 | 82 | MP (2x2) <br>
41 | 39 | 37 | 35 | 33 | 31 | MP (2x2) <br>
15 | 13 | 11| 9 | 7 | 5 | 3 | 1 <br>

By using maxpooling layer we have reduced the layer count from 200 to 27. That's great right ? 

<p align="center">
<img height="200" width="300" src="https://media.giphy.com/media/xT9IgzUuC5Ss6ZnTEs/giphy.gif">
</p>

Wait there is more!!!

Now, lets understand the concept of channels once again. In last post we have used single channel in input image for the calculations. Let's get practical and introduce our RGB channels of the image.

**How many kernels are we using in the network now?**

We have an image of size 400x400x3. Let's us assume we add 32 kernels in the first layer, 64 in second, 128 in thrid and so on.

Look at this animation and then we will look at the proper representation of our network.

<p align="center">
<img src="https://thumbs.gfycat.com/JointFewAmericancreamdraft-size_restricted.gif">
</p>

> Every kernel gives its own channel.
> Our kernels must have an equal number of channels as in the input channel.

So, our network will look like,

|Object Size|Kernel Size|Output Size|
|----|----|----|
|400x400x3|(3x3x3)x32|398x398x32
|398x398x32|(3x3x32)x64|396x396x64|
|396x396x64|(3x3x64)x128|394x394x128|
|394x394x128|(3x3x128)x256|392x392x256|
|392x392x256|(3x3x256)x512|390x390x512|
|MP||
|...|...||


Let's understand how convolution process is taking place using the above architecture.

If we have an object with 3 channels, there should be 1 kernels of size (3x3x3) where last index value 3 are channels of the kernel.

Look at image below, to understand how multi channels are handled.

![](https://miro.medium.com/max/1440/1*ciDgQEjViWLnCbmX-EeSrA.gif)

Do you see each kernel has it's own channel. Now we have another problem.

**How many parameters are we initializing ?**

Let's go through the network again,

|Object Size|Kernel Size|Output Size|Parameters|
|----|----|----|----|
|400x400x3|(3x3x3)x32|398x398x32|864|
|398x398x32|(3x3x32)x64|396x396x64|18432|
|396x396x64|(3x3x64)x128|394x394x128|73728|
|394x394x128|(3x3x128)x256|392x392x256|294912|
|392x392x256|(3x3x256)x512|390x390x512|1179648|
|MP|||
|195x195x512|(3x3x512)x512|193x193x512|2359296|
|...|...|...|...|


**How does increase in channel numbers affect the network and machine?**

From the above table, we have kernels as 32,64,128,256,512. So, how do I know whether my 512 channels are enough for my network to learn? Answer is we don't know, we need to experiment with these numbers.

Now, look at kernel size of 5th layer. We have (3x3x256)x512 as our kernel size. In 6th layer we have like 23M parameters. These increasing number of parameters can really slow down the networks learning process.

> The kernel size = number of learnable parameters

We have asked our network to learn 23M parameters just in 6th layer. Are you able to see what's happening. The more we add these parameters and ask our network to learn, the more it will slow down.

If you are very very rich person who can buy expensive GPUs, you can add any number of parameters in the network.

<p align="center">
<img src="https://media.giphy.com/media/JpG2A9P3dPHXaTYrwu/giphy.gif">
</p>

Training your network on K80 GPU will be slower than V100 GPU or any higher gpu model series.

**What's the solution?**

<p align="center">
<img src="https://media.giphy.com/media/3z3bxq78R9fVK/giphy.gif">
</p>

Stay tuned for the post!!!

For this post, we will continue with code now!!! Some hands on experience is necessary, too much theory !!! bruh... boring...

<p align="center">
<img src="https://media.giphy.com/media/bzE1WAm8BifiE/giphy.gif">
</p>

# Into the Code


```python
#import libraries

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (10,5)
```

- *datasets* will be used to download MNIST dataset which has been cleaned for us and provided by pytorch. 
- *transforms* are used to convert arrays to tensors which are used in pytorch framework. These can also be used to do some augumentation on the data. We will go through augumentation techniques in another post.

**GPU for training**

In order to use GPU, we need to identify and specify GPU as the device. Later, in training loop, we will load the data onto the device.


```python
import tensorflow as tf

device_name = tf.test.gpu_device_name()

try:
  print(f"Found GPU at : {device_name}")
except:
  print("GPU device not found.")
```


```python
import torch

if torch.cuda.is_available():
  device = torch.device("cuda")
  use_cuda = True
  print(f"Number of GPU's available : {torch.cuda.device_count()}")
  print(f"GPU device name : {torch.cuda.get_device_name(0)}")
else:
  print("No GPU available, using CPU instead")
  device = torch.device("cpu")
  use_cuda = False
```

Here just check how many gpu's do you have and what kind of gpu you are using as it affects the learning process of the network.

**Loading MNIST dataset**

Before creating any CNN network, we will first visualze and do some analysis on our MNIST dataset.

MNIST dataset contains 60,000 training and 10,000 test images. Each image is of size (28x28x1).

- We'll use a batch_size = 128 for training.
- The values 0.1307 and 0.3081 used for the Normalize() transformation below are the global mean and standard deviation for MNIST dataset.

Why are we normalizing?
<b>Geoffrey Hinton's</b> [learning](http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf) about gradient descent: 

> Going	downhill	reduces	the	error,	but	the direction	of	steepest	descent	does	not	point at	the	minimum	unless	the	ellipse	is	a	circle.


```python
torch.manual_seed(1)
batch_size = 128

kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
```


```python
mnist_trainset = datasets.MNIST(root="./data", train=True, download=True,
                                transform=transforms.Compose([
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.1307,), (0.3081,))
                    ]))

mnist_testset = datasets.MNIST(root="./data", train=False, download=True,
                               transform=transforms.Compose([
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.1307,), (0.3081,))
                    ]))
```


```python
train_loader = torch.utils.data.DataLoader(mnist_trainset,
                                          batch_size=batch_size, shuffle=True, **kwargs)

test_loader = torch.utils.data.DataLoader(mnist_testset,
                                          batch_size=batch_size, shuffle=True, **kwargs)
```

The above code will download the MNIST dataset, apply transforms and load the tensors into dataloader.


**Visualizing the dataset**

Let's see what our data looks like.

```python
examples = enumerate(train_loader)
batch_idx, (example_data, example_targets) = next(examples)
```

train data batch shape : [128, 1, 28, 28]

we have 128 images of size (128x128) in gray scale (no rgb channel).


```python
fig = plt.figure()
for i in range(6):
  plt.subplot(2,3,i+1)
  plt.tight_layout()
  plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
  plt.title(f"Ground Truth : {example_targets[i]}")
```

<p align="center">
<img src="https://github.com/myselfHimanshu/data-summit-blog/raw/master/images/cnn_blog_02/mnist_images.png">
</p>


**Building up the model**

In the below code, we will write what input size we are getting, what will be the output and what is the receptive field.

Things to keep in mind, we already saw what happens to receptive field size when applying kernel size 3x3 on object, everytime there is an addition of 2. 

But when there is max pooling layer in between, the receptive field doubles. The sentence is not completely true. We will see the effect on receptive field and derive a formula later, as it depends on stride, padding and other factors. So, just go through the post as it is for now. We will learn the concepts slowly.

I want you to understand what is happening in the network.

```python
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1) # input= [128,1,30,30], output = [128,32,28,28], rf = 3
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1) # input= [128,32,30,30], output = [128,64,28,28], rf = 5
        self.pool1 = nn.MaxPool2d(2, 2) # input= [128,64,28,28], output = [128,64,14,14], rf = 10
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1) # input= [128,64,16,16], output = [128,128,14,14], rf = 12
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1) # input= [128,128,16,16], output = [128,256,14,14], rf = 14
        self.pool2 = nn.MaxPool2d(2, 2) # input= [128,256,14,14], output = [128,256,7,7], rf = 28
        self.conv5 = nn.Conv2d(256, 512, 3) # input= [128,256,7,7], output = [128,512,5,5], rf = 30
        self.conv6 = nn.Conv2d(512, 1024, 3) # input= [128,512,5,5], output = [128,1024,3,3], rf = 32
        self.conv7 = nn.Conv2d(1024, 10, 3) # input= [128,1024,3,3], output = [128,10,1,1], rf = 34

    def forward(self, x):
        x = self.pool1(F.relu(self.conv2(F.relu(self.conv1(x)))))
        x = self.pool2(F.relu(self.conv4(F.relu(self.conv3(x)))))
        x = F.relu(self.conv6(F.relu(self.conv5(x))))
        x = F.relu(self.conv7(x))
        x = x.view(-1, 10) # [128,10]
        return F.log_softmax(x)
```

Now, as this is very simple network, we should get around 98% accuracy on test dataset even if I train on 1 epoch. But here, the network will behave strange. Find out why ? I have also included the link of code at the end of post.

**The architecture**

```python
from torchsummary import summary

model = Net().to(device)
summary(model, input_size=(1, 28, 28))
```

    ----------------------------------------------------------------
            Layer (type)               Output Shape         Param #
    ================================================================
                Conv2d-1           [-1, 32, 28, 28]             320
                Conv2d-2           [-1, 64, 28, 28]          18,496
             MaxPool2d-3           [-1, 64, 14, 14]               0
                Conv2d-4          [-1, 128, 14, 14]          73,856
                Conv2d-5          [-1, 256, 14, 14]         295,168
             MaxPool2d-6            [-1, 256, 7, 7]               0
                Conv2d-7            [-1, 512, 5, 5]       1,180,160
                Conv2d-8           [-1, 1024, 3, 3]       4,719,616
                Conv2d-9             [-1, 10, 1, 1]          92,170
    ================================================================
    Total params: 6,379,786
    Trainable params: 6,379,786
    Non-trainable params: 0
    ----------------------------------------------------------------
    Input size (MB): 0.00
    Forward/backward pass size (MB): 1.51
    Params size (MB): 24.34
    Estimated Total Size (MB): 25.85
    ----------------------------------------------------------------

torchsummary is very nice package that will give us output layer size and parameters information. You see there are total 6,379,786 learnable parameters. We will go around optimizing the code in future posts. 


**Training the model**


```python
from tqdm import tqdm #progress bar

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    pbar = tqdm(train_loader)
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device) #training of the device
        optimizer.zero_grad()
        output = model(data) # prediction
        loss = F.nll_loss(output, target) # calculate loss
        loss.backward() # backpropagtion step
        optimizer.step() # updating the parameters
        pbar.set_description(desc= f'loss={loss.item()} batch_id={batch_idx}')


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
```


```python
# SGD : stochastic gradient descent, lr:lerning_rate:0.01
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

for epoch in range(1, 2): # training network on 1 epoch
    train(model, device, train_loader, optimizer, epoch)
    test(model, device, test_loader)
```

You might be getting very low accuracy on test dataset. This shouldn't happen. Find the error above and comment below.

I have included colab notebook [link](https://gist.github.com/myselfHimanshu/b9f9f024c14eaa87a271172746b79eac). Run and play around.

Stay tuned, Happy Learning!!!

If you feel that I can provide you with value, I encourage you to connect with me, follow me, add me to your circles etc.

