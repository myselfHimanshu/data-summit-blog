+++
draft = false
date = 2019-02-22T22:22:22+05:30
title = "Sequence Model Part 2 : Word Embeddings"
slug = ""
tags = ["NLP","RNN"]
categories = []
math = "true"
+++

## Natural Language Processing and Word Embeddings

### Word Embeddings

**Word Representations**

Word Embeddings is the way of representing the words. It lets us understand the analogies between words like ("king" and "queen") or ("man" and "woman").

In the previous post we represented the words with a *one-hot vector*. One of the weakness of one-hot vector representation is that it treats a word as a thing and doesn't allow to generalize across the words.

The dot-product between two one-hot vectors is always zero. We can't get any similarity between the words using this representation for our other problems.

Solution is to learn  some featurized representation.

F/W|Man|Woman|King|Queen|Apple|Orange|
:-:|:-:|:-:|:-:|:-:|:-:|:-:|
Gender|-1|1|-0.95|0.97|0.00|0.01
Royal|0.01|0.02|0.93|0.95|-0.01|0.00
Age|0.03|0.02|0.7|0.6|0.03|-0.02
Food|0.02|0.01|0.02|0.01|0.95|0.97
..|..|..|..|..|..|..

We can come up with many features(the rows) let say we have 300 features, then this 300 dimension vector of value in columns becomes our representation.

To visualize this representation we use *t-SNE* algo to reduce the features to 2-dimension
![](https://github.com/myselfHimanshu/data-summit-blog/raw/master/images/part2_1.png)

We can see the related words are grouped together.

**Using Word Embeddings**

Given a named-entity recognition problem.
![](https://github.com/myselfHimanshu/data-summit-blog/raw/master/images/part2_2.png)

In this example *Sally Johnson* is a person name. And one way to know that it is person name and not organization is that *orange farmer* is a person. Suppose now we have our trained model using featurized word embeddings.

Now we get a new test example say *"Robert Lin is an apple farmer"*, the model will generalize that *apple* and *orange* are quite similar so *Robert Line* should be person's name. We can find some pre-trained word embeddings vectors online which we can be used in our own problem.

Transfer Learning and Word Embeddings:

- Learn word embeddings from a large text corpus (1-100B words) or download pre-trained embedding online.
- Transfer embedding to new task with the smaller training set (say, 100k words).
- Optional: continue to finetune the word embeddings with new data.
	- You bother doing this if your smaller training set (from step 2) is big enough.

The advantages of using word embeddings is that it reduces the size of the input. 10k one hot compared to 300 features vector.

**Properties of Word Embeddings**

It helps with analogy reasoning. Let's see the above table. We need to conclude :

`$$( Man : Woman ) :: ( King : ? )$$`

Get the vectors of  `$e_{man}, e_{woman}, e_{king}, e_{queen}$`

`$$e_{man} - e_{woman} \approx [-2, 0, 0, 0]$$
$$e_{king} - e_{queen} \approx [-2, 0, 0, 0]$$`

So the difference is about the gender in both. So to get this analogy we need to find a word which holds the equation true.

`$$find(w) = \underset{w\in W}{\operatorname{argmax}} (similarity(e_w,e_{king} - e_{man} + e_{woman}))$$`

Cosine Similarity :

`$$CosineSimilarity(u, v) = \frac {u . v} {||u||_2 ||v||_2} = cos(\theta)$$`

We can also use Euclidean distance as a similarity function.


### Learning Word Embeddings: Word2vec and GloVe

*How do we learn the embedding matrix?*

First look at this diagram.
![](https://github.com/myselfHimanshu/data-summit-blog/raw/master/images/part2_4.png)

Here we need to predict the next word so we use neural network to build the language model.

- `$O_i$` is one-hot vector
- `$E$` is featurized matrix of parameters
- `$e_j$` is the vector obtained by `$np.dot(E, O_i)$`
- These vectors are then feeded to a NN layer and then feeds to softmax which classifies the likelihood of words in vocab.
- The aim is to optimize the matrix `$E$` and the NN parameters. We need to maximize the likelihood to predict the next word given the context.

The context that we define to train the model can be different. For an example in this case we can give last 4 words and ask to predict the next word or we can give 2 words from right and 2 words from left and ask to predict the middle word or we can give nearby 1/2 words and the ask to predict the next word etc.

**Word2vec**

*Skip-grams*

Let say we have a sentence : *"I want a glass of orange juice to go along with my cereal"*. We choose a **context** and **target** word

Context|Target
:-:|:-:
orange|juice
orange|glass
orange|my

This is supervised problem and learning words in a given window size given context is hard as there can be many words that can come in that window. We want to learn this to get word embedding model.

*Model*

- Vocab-size = `V` =10k   
- Context word `c` and target word `t` and want to learn mapping of `c` to `t` .
- We get `$e_c$` by `$np.dot(E, O_c)$`
- We use softmax layer to get `P(t|c)` to get `$\hat{y}$`

Softmax :

`$$P(t|c) = \frac {e^{\theta_t^{T}e_c}}{\sum_{v=1}^V e^{\theta_v^{T}e_c}}$$`

`$\theta_t$` is parameter associated with output `t` i.e. what is a chance of `t` being a label.

- Will use the cross-entropy loss function.
- This is skip-gram model.
- Primary problem is computational speed. As we are computing the sum of vocab size in denominator every time we predict `t`.
- Solution to this is to use *Hierarchical softmax classifier*.
- Here, the softmax objective is slow to compute.

Word2vec paper includes two ideas of learning word embeddings. One is skip-gram model and another is CBoW (continious bag-of-words).

**Negative Sampling**

This is much more efficient learning algorithm that the above model. It does something similar to the skip-gram model.

Lets go with the same sentence : *"I want a glass of orange juice to go along with my cereal"*. We choose a **context** `c` and **target_word** `t` word with one more attribute saying if `c` and `t` are context pair `y` or not.

Context|Target_Word|Target
:-:|:-:|:-:
orange|juice|1
orange|glass|0
orange|my|0
orange|book|0

- We get positive example by using the same skip-grams technique, with a fixed window that goes around.
- To generate a negative example, we pick a word randomly from the vocabulary.
- This is supervised problem. Given a pair of words `c` and `t` predict if it is a pair `y`.
- apply the simple logistic regression model

`$$P(y=1|c,t) = sigmoid(\theta_c^{T}e_c)$$`

This will give us vocab_size binary classification which is cheaper than vocab_size softmax classification. Suppose we take only 5 samples for each context word, then I only have to train 5 for each context word which is efficient to train.

**GloVe Word Vectors**

Similar to the previous example, in here we calculate this for every pair, how related `c` and `t` are.

`$X_{ct}$` = number of times `t` appears in context of `c`

The model to minimize is defined:

`$$\sum_{c=1}^{V} \sum_{t=1}^{V} f(X_{c,t})(\theta_c^{T}e_t + b_c - b_t - \log X_{ct})^2$$`

- f(x) - the weighting term, used for many reasons which include:
	- The log(0) problem, which might occur if there are no pairs for the given target and context values.
	- Giving not too much weight for stop words like "is", "the", and "this" which occur many times.
	- Giving not too little weight for infrequent words.

### Applications using Word Embeddings

**Sentiment Classification**

Given a sentence or review we need to classify the ratings.

![](https://github.com/myselfHimanshu/data-summit-blog/raw/master/images/part2_5.png)

For this problem we might not have very large corpus but we can use pre-trained embedding vectors and it will work.

*Simple Sentiment classification model*

![](https://github.com/myselfHimanshu/data-summit-blog/raw/master/images/part2_6.png)

- The embedding matrix may have been trained on say 100B words.
- Dimension of word embedding is 300.
- We can use sum or average of all the words and then pass it to a softmax classifier.

This simple model will work for short sentences but doesn't take order of words in account which is very important. For example "Completely lacking in good taste, good service, and good ambience" has the word `good` 3 times but its a negative review.

*RNN for sentiment classification*

The many to one architecture:
![](https://github.com/myselfHimanshu/data-summit-blog/raw/master/images/part2_7.png)

This model will take the *lacking in good* into account and this model will also generalize better even if words weren't in our dataset.

*Problem of bias in word embeddings*

The problem here is suppose our model has learned some analogies and now we asked our model :

`$$( Man : Computer Programmer ) :: ( Woman : ? )$$`

ther were some results like

`$$( Man : Computer Programmer ) :: ( Woman : Homemaker )$$`
`$$( Father : Doctor ) :: ( Mother : Nurse )$$`

Which is just wrong. Word embeddings can reflect gender, ethnicity, age, sexual orientation, and other biases of text used to train the model. Learning algorithms by general is making an important decision and it mustn't be biased.

Addressing `gender` Bias:

- Identify the bias direction. `$e_{he} - e_{she}$`, `$e_{male} - e_{female}$` ...
- Choose some k differences and average them.
- We have found the bias direction(1D vector) and the non-bias direction(299D vector).
- Neutralize: For every word that is not definitional (girl, boy, he, she..), project to get rid of bias.
- Equalize Pairs: We want each pair to have difference only in gender.

## Problem Statements

*Coursera Assignment*

- <a href="https://github.com/myselfHimanshu/Coursera-DataML/blob/master/deeplearning.ai/Course5/Operations%2Bon%2Bword%2Bvectors%2B-%2Bv2.ipynb" target="_blank">Operation on Word Vectors and Debiasing.</a>
- <a href="https://github.com/myselfHimanshu/Coursera-DataML/blob/master/deeplearning.ai/Course5/Emojify%2B-%2Bv2.ipynb" target="_blank">Emojify.</a>
