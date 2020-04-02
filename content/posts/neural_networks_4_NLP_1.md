+++
draft = false
date = 2020-02-20T13:33:03+05:30
title = "Intro to Neural Networks for NLP"
slug = ""
tags = ["NLP","DeepLearning"]
categories = []
math = "true"
+++

### Overview

Language is HARD and COMPLICATED. There are some techniques which we can use to extract information from the text. There are things that needs to be handled to be able to understand any text sentence like morphology, syntax, word knowledge, discourse, pragmatics, multi linguality etc. For these hard things we use Neural Networks.

In this blog, we will learn about some concepts to do such things.

**Problem Statement** : Sentiment Classification

A very basic problem to start with.

Given a sentence/text as an input, we need to classify the sentiment label defined below.

Sentence|very good|good|neutral|bad|very bad
:-:|:-:|:-:|:-:|:-:|:-:|
I hate this movie|0|0|0|0|1
I love this movie|1|0|0|0|0

To classify these sentences, there can be different approaches and these approaches have their own cons and pros.

### First Try : BOW (Bag of words)

The above statement : `I hate this movie` can be divided into individual lexicons. Each lexicon will have its own score value.<br>

For an example, lexicon `hate` can have a score of lets say -2.0, where as `bad` can be -1.5

Each word has its own 5 elements corresponding to [very good, good, neutral, bad, very bad]

`hate` will have high value for `very bad`

Then each lexicon vector is added to get a final output scores. We can apply a softmax function to get the probability distribution of the score vector and the max score index will indicate the sentiment of the sentence.

This can be used as our baseline model before going on to create or work on some more complex model.

The gist of BOW model is, you throw all the words in the vocabulary into a bag. These words will have a vector representation. Given a statement, get all the word vectors present in the sentence, calculate the score, apply some probability function and predict the label.

**Where does this breaks :**

Consider these two sentences:

- I don't love this movie.
- There's nothing I don't love about this movie.

The second statement have a positive sentiment. But if we apply BOW, this might fail as there are more negative lexicons in the statement which may lead to a negative sentiment.

So now we know where it breaks. We need some sort of combination of features, like feature vector of combination of `nothing` with `don't love` maybe.

To get these combination features we use neural networks. We will take these lexicons vector representation and feed it into some complicated function to extract combination features(neural nets) and then get the predictions.

### Second Try : CBOW (continuous bag of words)

The concept here is the same as BOW model but instead of just adding all the vector scores, we will multiply the score vector with some weight matrix, add some bias and get the prediction. <br>
Each vector has some features like (is this is a pronoun, is this a positive word etc). Still there is not combination included in this model but dimensions are reduced.

### Second Try Version 2 : Deep CBOW (continuous bag of words)

There is a slight addition into the CBOW model. After taking all the lexicons vectors, we multiply it with some weights matrix, add the vectors and then feed into some non-linear function which will allow us to use the combination of features.

Before we go into deeper concepts, let's try to understand the NN.

### Some notes about Neural Nets

You can think Neural Networks as Computational Graphs. These Networks are considered Universal Function Approximators.<br>
Original Motivation : Neurons in a brain.

Simple Concept about any Neural Network model:<br><br>

<div style="display: flex; justify-content: center;">
  <img src="https://github.com/myselfHimanshu/Portfolio-Website/blob/master/images/neural-net-1.png?raw=true">
</div>

<br>

What we need is take input vector and apply some function approximation to make input vector move closer to the predicted vector.

This is where loss function comes into picture. You have this loss function which tells you how far is your input vector to the predicted vector.

We have studied differentiation in high school. Same concept is applied over here (don't cry over the saying that learning differentiation in school was waste). We use differentiation to get the slope or in which direction where we need to send the input vector to make it closer to the predicted vector.

These function approximation are the non-linear activation function (check the figure down) we use in computational graph or say neural networks models.

**Why are we using these non-linear function?**

A data can be separated into two categories. One which can be linear separable and other not. For simple linear separable problems one can use any of LR, Perceptron, Decision trees etc. (used to draw linear decision boundaries) and get the output easily.

The linear separable data is straight forward and a linear function is just a polynomial of degree of 1. These function have less power to learn complex functional mapping from data.

Some functions are defined below:

![](https://github.com/myselfHimanshu/Portfolio-Website/blob/master/images/non-linear-functions.png?raw=true)

Non-linear functions are those which have degree more than one and they have a curvature when we plot a non-linear function. The data is not always straight forward, there are different patterns, it can be complex and complicated.

We need to apply these non-linear functions f(x)(also termed as activation functions (don't get confused)) to make our model more powerful and make them learn something complex and complicated from data and represent non-linear complex arbitrary functional mappings between inputs and outputs.

**Which one to use?**

We will discuss about these in detail in later posts. Just for a gist nowadays we use ReLu which should only be applied to the hidden layers. And if your model suffers form dead neurons during training we should use leaky ReLu.

Sigmoid and tanh are not being used nowadays due to the vanishing Gradient Problem which causes a lots of problems to train, degrades the accuracy and performance of a Neural Network model.

**Computational Graph**

A Computational Graph is a directed and cyclic graphs of differentiable and sub-differentiable functions.

This is a computational graph for an expression :

$$ y = x^T Ax + bx + c $$

![](https://github.com/myselfHimanshu/Portfolio-Website/blob/master/images/computational-graph.png?raw=true)

- `X` : A node can be {tensor, matrix, vector, scalar etc}
- An edge represents function argument.
- An edge represents function argument.
- A node with incoming edge is a function that edge's tail node.
- A function can be nullary, unary, binary, ..., n-ary. In above image $ f(U,V) $

### Things to know when creating or defining any model
- Graph Construction
- Forward Propagation : In topological order, compute value of a node given it's input.
- Back Propagation : Calculation of derivative of the parameters w.r.t to the final output values.
- Parameter Update : Move the parameters in the direction of this derivative.

Don't worry, computing some of these will be taken care by the framework until you want to write it from scratch.

### Frameworks

Static Frameworks : Static declaration of computational graph. (theano, caffe, mxnet, tensorflow)
Dynamic Frameworks : Dynamic declaration of computational graph. (dynet, chainer, pytorch)

Will add code link later for what we have learned.

**Assignments**<br>
1. <a href="https://gist.github.com/myselfHimanshu/92c7a5d0352364accf3a1959338fbfe9" target="_blank">BOW Model in pytorch</a>.


References:

- http://phontron.com/class/nn4nlp2019/index.html
