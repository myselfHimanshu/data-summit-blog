+++
draft = false
date = 2019-11-03T21:46:34+05:30
title = "Sequence Model Part 1 : RNNS"
slug = ""
tags = ["NLP","RNN","DeepLearning"]
categories = []
math = "true"
+++

## Recurrent Neural Networks

Some Examples

|Example | X (input) | Y(output) | Type |
|:----:|:----:|:----:|:----:|
|Speech Recognition|wave sequence|text sequence|sequence to sequence|
|Music Generation|nothing or integer|wave sequence|one to sequence|
|Sentiment Classification|text sequence|integer(label)|sequence to one|
|Machine Translation|text sequence|text sequence|sequence to sequence|
|Video Activity Recognition|video frames|label(activity)|sequence to one|

All these problems have different types of input and output formats.

### Named Entity Recognition

Example |||||||||
:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
X|Harry|Potter|and|Hermoine|Granger|invented|new|spell
Y|1|1|0|1|1|0|0|0

This is a problem where every `$X^{(i)<t>}$` element has `$Y^{(i)<t>}$` as ouput element.


### Representing Words
We need a vocabulary which will contain all the unique words from the corpus and an index will be assigned to each element.
The above given example was only one vector. Suppose we have a whole novel, we'll create the vocabulary dictionary.

Vocabulary Dictionary : vocabulary of top 10000 words from novel most occuring words sorted.

Words|a|..|aron|...|...|harry|..|..|potter|..|zulu
:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
Index|1|...|2|...|...|3069|...|...|7867|...|10000

We can represent each word as *one-hot* representation. Where each word vector will be of size of length of vocabulary and the index where word occurs will be 1 and other values will be 0.

Vector of harry `$X^{<1>}$` will be

Index_Words|a|..|aron|...|...|harry|..|..|potter|..|zulu
:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
'Harry' Vecctor|0|...|0|...|...|1|...|...|0|...|0

Vector of potter `$X^{<2>}$` will be

Index_Words|a|..|aron|...|...|harry|..|..|potter|..|zulu
:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
'Potter' Vecctor|0|...|0|...|...|0|...|...|1|...|0


If we have a new word in a sentence but we don't have it in vocabulary, we can add new term in our vocab say \<UNK\> (UNKNOWN) and assign the value at its index.

*The aim is to build a model to learn the mapping between X and Y.*

### Recurrent Neural Network Model

*Why not a standard Network?*

1. Inputs and outputs can have different lenghts in different examples.<br />
Padding can be used, but it is not a good solution.
2. Doesn't share features learned across different positions of text.<br />
Using feature sharing like in CNNs can reduce number of parameters in the model.

An RNN architecture where:

- `$Tx==Ty$` same length sequence.
- `$a^0$` is initialized with zeros, can be random.
![](https://github.com/myselfHimanshu/Portfolio-Website/blob/master/images/rnn1.png?raw=true)

- In here for the prediction of `$y^{<3>}$`, model will use not only `$X^{<3>}$` but also the information that it is getting from the previous two elements i.e `$X^{<2>}$` and `$X^{<1>}$`.
- In RNN, the current prediction `$y^{<t>}$` will only depend on previous inputs but not the future inputs. (This case will be solved by *Bidirectional RNN*)

**Forward Propogation**

RNN Unit :
![](https://github.com/myselfHimanshu/Portfolio-Website/blob/master/images/rnn1_1.png?raw=true)

Notation :

`$$a^{<t>} = g(W_{aa}a^{<t-1>} + W_{ax}x^{<t>} + b_a)$$
$$y^{<t>} = g(W_{ya}a^{<t>} + b_y)$$`

The activation function of `$a$` is usually *tanh* or *relu* and `$y$` depends on the task, some activation functions like *sigmoid* or *softmax*.

**Backward Propogation through time**

Deep learning frameworks do backpropogation automatically for us. The architecture looks like this :
![](https://github.com/myselfHimanshu/Portfolio-Website/blob/master/images/rnn2.png?raw=true)

Here, `$W_a, b_a, W_y, b_y$` are shared across each element of the sequence.<br />

The loss here that we optimize is:

`$$L(y,\hat{y}) = \sum_{t=1}^T L^{<t>}(y^{<t>},\hat{y}^{<t>})$$
$$L^{<t>}(y^{<t>},\hat{y}^{<t>}) = -y^{<t>}log(\hat{y}^{<t>}) - (1-y^{<t>})log(1-\hat{y}^{<t>})$$`

Each loss is backpropogated in opposite direction of forward propogation steps and the parameters are updated by gradient descent.

### Different types of RNNs

![](http://karpathy.github.io/assets/rnn/diags.jpeg)

The architecture that we have described before is *many to many*. Other example with its architecture are mentioned above in table.

![](https://github.com/myselfHimanshu/Portfolio-Website/blob/master/images/rnn3.png?raw=true)

In above image there are two architecture descibed.

1. One to Many : Music Generation. We are feeding the predicted ouput from the first element to the input of second element.
2. Many to Many : Machine Translation. Here we can have different `$T_X, T_Y$` (length of input `$X$` and `$Y$` respectively). There are *encoder* and *decoder* part to this architecture, where output of encoder is feeded to decoder to generate the ouputs. And encoder and decoder have different weights matrices.

### Language Model and Sequence Generation

*What is language model?*

1. The apple and **pair** salad.
2. The apple and **pear** salad.

- The goal is to differentiate the two sentences in speech recognition probelem.
- These two sentences sounds the same. We want the model to give us the second statement.

So the language model predicts on the basis of likelihood or probability of the sentence.

*How do build this language model?*

- Get the Training set: large corpus.
- Tokenize each sentence, get the vocabulary and convert each element to one-hot vector representation.
- Add <EOS>(eod of statement) as token to every example to know the end of a example. And add <UNK>(unknown) and <EOS> token both to the vocabualry.

For an example, *Cats average 15 hours of sleep a day <EOS>*

![](https://github.com/myselfHimanshu/Portfolio-Website/blob/master/images/rnn4.png?raw=true)

- So given *Cats average 15* what is the probability the word is *hours*.
- To get a probability of a given sentence

`$$P(y^1, y^2, y^3) = P(y^1) * P(y^2|y^1) * P(y^3|y^2,y^1)$$`

### Sample Novel Sequences

- What we do in here is first we train suppose all the sentences of Harry Potter books.
- Then we randomly select a word let say one of the top word and feed that into next timestamp and then sample a new word from the prediction we got.
- Use the probabilities output by the RNN to randomly sample a chosen word for that time-step as `$\hat{y}^{<t>}$`. Then pass this selected word to the next time-step.
See this image here

![](https://github.com/myselfHimanshu/Portfolio-Website/blob/master/images/rnn5.png?raw=true)

- We feed in `$a^{<0>} = zeros$` and `$x^{<1>} = zeros$` vector.
- We will choose a prediction randomly from distribution obtained by `$y^{<1>}$`. Let assume that the word is *the*.
- We pass the last predicted output and activation to the next timestamp and again will choose a prediction randomly.
- Repeat the steps until we find *\<EOS\>* token.

Now we have our own series.

### Vanishing Gradients in RNN

One of the problem with RNN is *vanishing gradient*. Suppose for an example,<br />
"*The cat, which actually ate ...... {100}, was full*" <br />
"*The cats, which actually ate ......{100}, were full*" <br />

Here we need to remember the dependencies of *cat -> was* and *cats -> were* the singular/plural form for long time. This long sentences can be interpreted as DEEP RNN and here the gradients of cat/cats will no effect on was/were. While backpropogating mutilplying the fractions can lead to vanish the gradient while multiplication of large numbers explodes the gradient where weights values becomes `NaN` which is also called gradient exploding problem.

So RNNs are not good in learning *long-term dependencies*. Exploding gradient can be solved by *gradient clipping*.

### Gated Recurrent Unit (GRU)

This is updated version of RNN which solves *long-term dependencies* and *vanishing gradient* problem.

We will go step by step of this GRU Unit (*Simplified Version*).<br />

- Our example : "*The cat, which actually ate ...... {100}, was full*"
- The unit will have a new variable `$C = memory cell$`. This memory cell will tell us whether the cat was singular or plural. In other words what to memorize and what to not.
- At timestamp `t` -> `$C^{<t>} = a^{<t>}$`
- At every timestamp we will compute a candidate for the memory cell.

`$$\tilde{C}^{<t>} =  tanh(W_C[C^{<t-1>}, X^{<t>}] + b_C)$$`

- Then there is an update gate which have value between 0 and 1

`$$\Gamma_U =  sigmoid(W_U[C^{<t-1>}, X^{<t>}] + b_U)$$`

- We are thinking to update `$C^{<t>}$` with `$\tilde{C}^{<t>}$` and the gate `$\Gamma_U$` will tell us whether to update or not.

`$$C^{<t>} = (\Gamma_U * \tilde{C}^{<t>}) + (1-\Gamma_U) * C^{<t-1>}$$`

The GRU Unit:

![](https://github.com/myselfHimanshu/Portfolio-Website/blob/master/images/rnn6.png?raw=true)

Here the shaded box is our last equation `$(\Gamma_U * \tilde{C}^{<t>}) + (1-\Gamma_U) * C^{<t-1>}$`

Because the `$\Gamma_U$` is usually a small number like 0.00001, it doesn't suffer from vanishing gradient as in this case `$C^{<t>} = C^{<t-1>}$`

*GRU Unit (Full Version)*

Notation :

`$$\tilde{C}^{<t>} =  tanh(W_C[\Gamma_r * C^{<t-1>}, X^{<t>}] + b_C)$$
$$\Gamma_U =  sigmoid(W_U[C^{<t-1>}, X^{<t>}] + b_U)$$
$$\Gamma_r =  sigmoid(W_r[C^{<t-1>}, X^{<t>}] + b_r)$$
$$C^{<t>} = (\Gamma_U * \tilde{C}^{<t>}) + (1-\Gamma_U) * C^{<t-1>}$$`


This `$\Gamma_r$` can be interpreted as relevance, how relevant is `$C^{<t-1>}$` to compute `$\tilde{C}^{<t>}$`.


### Long Short Term Memory (LSTM)

The LSTM is a variation of the same theme as GRU but with an additional *forgot* gate. This also solves *long-term dependencies* and *vanishing gradient* problem and is more powerful than GRUs.

The LSTM Unit:
![](https://github.com/myselfHimanshu/Portfolio-Website/blob/master/images/rnn7.png?raw=true)


Notations:

`$$C^{<t>} != a^{<t>}$$
$$\tilde{C}^{<t>} =  tanh(W_C[a^{<t-1>}, X^{<t>}] + b_C)$$
$$\Gamma_U =  sigmoid(W_U[a^{<t-1>}, X^{<t>}] + b_U)$$
$$ \Gamma_f =  sigmoid(W_f[a^{<t-1>}, X^{<t>}] + b_f) $$
$$ \Gamma_o =  sigmoid(W_o[a^{<t-1>}, X^{<t>}] + b_o) $$
$$ C^{<t>} = (\Gamma_U * \tilde{C}^{<t>}) + (\Gamma_f * C^{<t-1>})  $$
$$ a^{<t>} = \Gamma_U * C^{<t>} $$`


Here we have 3 gates `$\Gamma_U$` as *update gate* ,  `$\Gamma_f$` as *forget gate* and `$\Gamma_o$` as *output gate* <br />
One of the advantages of GRU is that it's simpler and can be used to build much bigger network but the LSTM is more powerful and general.

### Bidirectional RNNs

Suppose for an example,<br />
*He said,"Teddy bears are on sale"* <br />
*He said,"Teddy Roosevelt was a great president"* <br />

If we look just first three words, we won't be able to say if *teddy* is a toy or person. This is why we need the information from the future elements also.

The architecture:
![](https://github.com/myselfHimanshu/Portfolio-Website/blob/master/images/rnn8.png?raw=true)

- BiRNN is an acyclic graph.
- The prediction `$\hat{y}^{<t>} = g(W_y[\overrightarrow{a}^{<t>}, \overleftarrow{a}^{<t>}] + b_y)$`
- The disadvantage of BiRNNs is that you need the full sentence before you make any predictions. So it is not suitable for live speech recognition.

### Deep RNNs

The architecture with 3 stacked layers
![](https://github.com/myselfHimanshu/Portfolio-Website/blob/master/images/rnn9.png?raw=true)

In feed forward DNNs, there could be 100 or 200 layers. In DRNNs stacking 3 layers is already considered deep and expensive to train.

## Problem Statements

*Coursera Assignment*

- <a href="https://github.com/myselfHimanshu/Coursera-DataML/blob/master/deeplearning.ai/Course5/Building%2Ba%2BRecurrent%2BNeural%2BNetwork%2B-%2BStep%2Bby%2BStep%2B-%2Bv3.ipynb" target="_blank">Building Neural Network Step by Step.</a>
- <a href="https://github.com/myselfHimanshu/Coursera-DataML/blob/master/deeplearning.ai/Course5/Dinosaurus%2BIsland%2B--%2BCharacter%2Blevel%2Blanguage%2Bmodel%2Bfinal%2B-%2Bv3.ipynb" target="_blank">Dinosaur Island - Character level language Modeling.</a>
- <a href="https://github.com/myselfHimanshu/Coursera-DataML/blob/master/deeplearning.ai/Course5/Improvise%2Ba%2BJazz%2BSolo%2Bwith%2Ban%2BLSTM%2BNetwork%2B-%2Bv3.ipynb" target="_blank">Jazz Improvisation with LSTM.</a>
