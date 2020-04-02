+++
draft = true
date = 2020-03-29T22:22:22+05:30
title = "Convolutional Neural Network Part 2 : Neural Architecture"
slug = ""
tags = ["CNN"]
categories = []
math = "true"
+++

In this tutorial, we will be training the network on MNIST dataset. 

We'll try to understand each and every step, what is input? and what will be the output? 

The most important thing to understand in deep learning is what goes in and what comes out, with the understanding of what the network is doing inside that black box. Here black box refers to the hidden layers of DNN. What is it seeing first and at the end of the training the model.

The below architecture is not meant to give us very good score. In this series we'll only focus on **what**.

Ready ? Let's jump in.

![](https://media.giphy.com/media/11OWKkvYUmZQOs/giphy.gif)


### Basic Concepts

To understand the basic terminologies of CNN building blocks. Kindly go through this post.

[CNN Part 1 : Basic Concepts](https://myselfhimanshu.github.io/posts/cnn_01/)

## Neural Architecture : Part II

In the below section we are introducting two main concept. 

*Max-pooling* and *receptive field.*

Now, in the last post we have calculated how many layers we have to use if we use 3x3 kernel on a 401x401 image size with stride=1 and padding=0?

The answer was 200.

Before jumping further, let me tell you **what padding is?**

You can think padding as extra rows and columns of pixels that are applied around a feature map.

![](https://lh3.googleusercontent.com/proxy/MoeTZDgmtTWfZ0BNFjBrFbkmURWFhn6Y_vBo_FiPc07sjCsxEF-yrwZN1sG5ZhoUP_kBOMtImaCtXFhohVDrHAGJWnU)

In the above image, we have a feature map of 6x6, if we apply a padding=1, we add a row and a column around that feature map. This feature map will result in size of 8x8 now on which we apply convolution.

It is not necessary to apply padding with value 0, as shown in the image. We will see what padding should we apply such that we can gain more information from around the corners of the feature map.

Now coming back to our layers concept.

**Do we really need 200 layers in our network?**

The answer is No.

We need our model to learn fast and learn accurate, for which we should built an architect wherein our last layer's receptive field should be the whole object. What I mean is, we want our network's receptive field to slowly increase as we add layers. Before taking any decision, the whole image needs to be processed.

> Refer to Building blocks of Convolutional Neural Network section of [previous post](https://myselfhimanshu.github.io/posts/cnn_01/).

To reduce number of layers, one technique to downsample a feature map is MaxPooling.

![](https://cdn-images-1.medium.com/freeze/max/1000/1*ghJyfuw-9a5esjJqGuBggA.jpeg?q=20)

It is clear what is happening from the above image.

If we look at first row, applying a max-pool layer of size 2x2 on 4x4 feature map with stride equal to the size of the layer gives 2x2 feature map.

**MaxPooling** is used for dimension reduction. It is a simple layer where no learning happens.

It helps in reducing the number of learned parameters, which helps in reducing the computation and memory load.

> We need to take care of where to apply max pooling layer in the network.

**Why is above line important?**

Let's work on an image of 4.

|image|apply maxpool|apply maxpool again|
|-----|------|------|
|![](https://github.com/myselfHimanshu/data-summit-blog/raw/master/images/cnn_blog_02/maxpool_org.png)|![](https://github.com/myselfHimanshu/data-summit-blog/raw/master/images/cnn_blog_02/maxpool_2.png)|![](https://github.com/myselfHimanshu/data-summit-blog/raw/master/images/cnn_blog_02/maxpool3.png)|

The above process is just applying maxpool again and again on the same image. This doesn't happen in the network. This is a concept that I am explaining you about where to use maxpool layer.

Now, if we apply one max pool layer of size 2x2 over first image, it will result in second image and if we apply another max pool layer we get image 3.

Now ask what is your network learning if I do this?

It might be learning bananas for minions.

![](https://media.giphy.com/media/ZqlvCTNHpqrio/giphy.gif)

So we need to really careful, where should I apply the maxpool layer.

> Never use max pooling close to your output/prediction layer.

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

400 | 398 | 396 | 394 | 392 | 390 | MP (2x2) <br>
195 | 193 | 191 | 189 | 187 | 185 | MP (2x2) <br>
92 | 90 | 88 | 86 | 84 | 82 | MP (2x2) <br>
41 | 39 | 37 | 35 | 33 | 31 | MP (2x2) <br>
15 | 13 | 11| 9 | 7 | 5 | 3 | 1 <br>

By using maxpooling layer we have reduced the layer count from 200 to 27.

Now, lets understand the concept of channels.

**How many kernels are we using in the network?**

We have an image of size 400x400x3. Let's us assume we add 32 kernels in the first layer, 64 in second, 128 in thrid and so on.

Look at this animation and then we will look at the proper representation of our network.

![](https://thumbs.gfycat.com/JointFewAmericancreamdraft-size_restricted.gif)

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

If we have an object with 3 channels, there should be 1 kernels of size (3x3x3) where last index value 3 is channels of a kernel which gives outputs.

Look at image below, to understand how multi channels are handled.

![](https://miro.medium.com/max/1440/1*ciDgQEjViWLnCbmX-EeSrA.gif)

While building a network be careful of receptive field. If the receptive field of the output is too small, ask yourself a question, is my network really learning something?

**How does the channels size increases per layer and what would be it's effect on the network or my machine ?**

From the above table, in 5th layer we get 512 channels. So, how do I know whether my 512 channels are enough for my network to learn? Answer is we don't know, we need to experiment with these numbers.

As this 512 is a large number, a machine which is capable of handling it will only be able to do it. If I ask you to run on say T4, your machine will take more time to train your network as compared to say P100 or any higher gpu model series.

This number depends on how complex is the dataset? like in medical use cases, we might need 1024 channels and it might still be not enough.

There is another problem in the above network. Check the table once again.

**How many parameters are we initializing ?**

More the number of parameters initialized slower will the training time of the network. Let's again go through the network,

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

Do you see the number of parameters that are increasing? 



In 6th layer we have like 23M parameters. These increasing number of parameters can really slow down the networks learning process.

What's the solution? 

![](https://media.giphy.com/media/3z3bxq78R9fVK/giphy.gif)

Stay tuned for the post!!!

For this post, we will continue with code now!!! Some hands on experience is necessary, too much theory !!! bruh... boring...

![](https://media.giphy.com/media/bzE1WAm8BifiE/giphy.gif)


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

### GPU for training 

In order to use GPU, we need to identify and specify GPU as the device. Later, in training loop, we will load the data onto the device.


```python
import tensorflow as tf

device_name = tf.test.gpu_device_name()

try:
  print(f"Found GPU at : {device_name}")
except:
  print("GPU device not found.")
```

    Found GPU at : /device:GPU:0



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

    Number of GPU's available : 1
    GPU device name : Tesla T4


### Loading MNIST dataset

Before creating any CNN network, we will first visualze and do some analysis on our MNIST dataset.

MNIST dataset contains 60,000 training and 10,000 test images. Each image is of size (28x28x1).

We'll use a batch_size of 128 for training.
The values 0.1307 and 0.3081 used for the Normalize() transformation below are the global mean and standard deviation for MNIST dataset.

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


```python
len(mnist_trainset), len(mnist_testset)
```

    (60000, 10000)



### Visualizing the dataset

Let's see what our data looks like.


```python
examples = enumerate(train_loader)
batch_idx, (example_data, example_targets) = next(examples)
```


```python
example_data.shape
```
    torch.Size([128, 1, 28, 28])


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


![](https://github.com/myselfHimanshu/data-summit-blog/raw/master/images/cnn_blog_02/mnist_images.png)


### Building up the model


```python
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1) #input size, output size, rf
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv5 = nn.Conv2d(256, 512, 3)
        self.conv6 = nn.Conv2d(512, 1024, 3)
        self.conv7 = nn.Conv2d(1024, 10, 3)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool2(x)
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        x = x.view(-1, 10)
        return F.log_softmax(x)
```

### The architecture


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


    /usr/local/lib/python3.6/dist-packages/ipykernel_launcher.py:25: UserWarning: Implicit dimension choice for log_softmax has been deprecated. Change the call to include dim=X as an argument.


### Visualization of kernels and channels


```python
model.conv1.weight.data.cpu().numpy().shape
```




    (32, 1, 3, 3)




```python
from torchvision.utils import make_grid

def visualize_tensor(tensor):
  tensor = tensor - tensor.min()
  tensor = tensor / tensor.max()
  img = make_grid(tensor)
  plt.imshow(img.permute(1,2,0))
```

### Training the model


```python
from tqdm import tqdm
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    pbar = tqdm(train_loader)
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
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
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

for epoch in range(1, 2):
    train(model, device, train_loader, optimizer, epoch)
    
    k1 = model.conv1.weight.data.cpu().clone()
    visualize_kernel(k1)

    activation = {}
    def get_activation(name):
      def hook(model, input, output):
        activation[name] = output.detach()
      return hook

    model.conv1.register_forward_hook(get_activation('conv1'))
    o1 = activation['conv1']
    visualize_tensor(o1)


    # test(model, device, test_loader)
```

      0%|          | 0/469 [00:00<?, ?it/s]/usr/local/lib/python3.6/dist-packages/ipykernel_launcher.py:25: UserWarning: Implicit dimension choice for log_softmax has been deprecated. Change the call to include dim=X as an argument.
    loss=0.6927734017372131 batch_id=468: 100%|██████████| 469/469 [00:12<00:00, 36.63it/s]



    ---------------------------------------------------------------------------

    KeyError                                  Traceback (most recent call last)

    <ipython-input-64-fc4779b0fb46> in <module>()
         14 
         15     model.conv1.register_forward_hook(get_activation('conv1'))
    ---> 16     o1 = activation['conv1']
         17     visualize_tensor(o1)
         18 


    KeyError: 'conv1'



![png](output_34_2.png)



```python

```
