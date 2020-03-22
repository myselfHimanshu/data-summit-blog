+++
draft = false
date = 2020-02-23T22:22:22+05:30
title = "Sequence Model Part 3 : Sequence models & Attention mechanism"
slug = ""
tags = ["NLP","RNN"]
categories = []
math = "true"
+++

### Various Sequence to Sequence Architecture

**Basic Model**

Let's start with a machine translation model of converting french to english:
![](https://github.com/myselfHimanshu/data-summit-blog/raw/master/images/part3_1.png)

Given a sequence `X` we need the output `y`. Here we have a network called as `encoder` with *gru* or *lstm* blocks which feeds in french words one at a time and outputs a vector that will represent our input french sentence. This vector can be feeded to another network called as `decoder` and can be trained to output the translated sentence one word at a time.

![](https://github.com/myselfHimanshu/data-summit-blog/raw/master/images/part3_2.png)

This kind of network can also be used for *Image Captioning* task. Given an image of cat `X` as input, predict the caption `y` "A cat sitting on a chair".

![](https://github.com/myselfHimanshu/data-summit-blog/raw/master/images/part3_3.png)

We can first train our CNN and then feed the last layer as input to our decoder network to predict the caption.

**Picking the most likely sentence**

How is our *language model* different from this *machine translation* model.

![](https://github.com/myselfHimanshu/data-summit-blog/raw/master/images/part3_4.png)

The `decoder` network is pretty much same as our language model except the `$a^{<0>}$` part where we are using the `encoder` network which figures out the way of representing our input sentence. This can also be called as `conditional language model`.

- In language model :

`$$P(y^{<1>},y^{<2>},...,y^{<T_y>})$$`

- In machine translation :

`$$P(y^{<1>},y^{<2>},...,y^{<T_y>} | x^{<1>},x^{<2>},...,x^{<T_x>})$$`

We don't want a randomly generated sentence. Sometimes we would get good sentences and sometimes bad. That's why we need to find `y` that maximizes the above probability.

*Idea: Use greedy search*

We are talking about decoder part.

- Pick the word with the best probability and repeat until we get `<EOS>`.
With this we can get "Jane is going to be visiting Africa this September". Note this method picks up word by word.

**Beam Search**

Beam search is the most widely used algorithm to get the best output sequence.

1. Pick the first word. Pick first `B`(say 3) best alternatives. Suppose, `P(y^{<1>}|x}`  gives us {in, jane, september} as possibilities.
2. Now considering these 3 words, the model will pick up the second word based on liklihood. Here we are calculating,

`$$\prod_{t=1}^{T} P(y^{<t>}|x, y^{<1>},...,y^{<t-1>})$$`

And repeat the second step and evaluate the probabilities word by word. Here we need to keep track of the `P` for each sentences of each length and after `i` interations pick the best (remember we need the output `y` which maximizes the probability of the whole sentence)and continue.

If `B`=1, this will become a greedy search.

**Refinements to Beam Search**

First thing is Length optimization. We were trying to maximize the probablity by muliplying. But multiplying these small fractions will cause a `numerical overflow!`. So instead of multiplying we use `summing logs` and normalize it.

`$$\frac{1}{T_{y}}\sum_{t=1}^{T_y} \log P(y^{<t>}|x, y^{<1>},...,y^{<t-1>})$$`

*The second thing is choosing best B?*

- The larger B -> larger possibilities -> better are the results -> but more computationally expensive.
- In practice, production systems B=10, B=100, B=1000 might not be uncommon but it is domain dependent.
- Beam Search runs faster than BFS and DFS but is not guaranteed to find exact solution.

**Error Analysis in beam search**

Let's start with an example:

- X : "Jane visite l’Afrique en septembre."
- y : "Jane vists Africa in September."
- `$\hat{y}$` : "Jane visited Africa last September."

Our model has two main components:
1. RNN network
2. Beam Search Algorithm

As our algorithm doesn't predict right. How do we know if it's RNN or Beam Search we should work on?

Solution: Here we will calculate `$P(y|x)$` and `$P(\hat{y}|x)$`. Cases : <br />
If `$P(y|x)$` > `$P(\hat{y}|x)$`:

- This means beam search chose `$\hat{y}$` but `$y$` attains the higher probability.
- Beam Search is at fault.

If `$P(y|x)$` <= `$P(\hat{y}|x)$`:

- `$y$` is better translation than `$\hat{y}$` but RNN predicted probability of `$\hat{y}$` greater than `$y$`.
- RNN model is at fault.

Then we can calculate this for every example in our dev set and get the value counts of `Beam search fault` and `RNN model fault` and work on model accordingly.

**Bleu Score**

One challenge is that there can be multiple possible good translation for a given sentence. So how do we evaluate the result?

Solution: "BLEU score". BLEU stands for bilingual evaluation understudy.

Let's understand with an example:

- X : "Le chat est sur le tapis."
- Y1 : "The cat is on the mat."
- Y2 : "There is a cat on the mat."

The intuition is that we are going to look at the predicted statement and see if the types of words appear in any of the human generated statement. Suppose our predicted statement is :

- `$\hat{y}$` : "the the the the the the the."

One way to measure is to calculate the precision. In the above predicted statement there are 7 words and each of then either comes in Y1 or Y2. So, `$precision = \frac{7}{7}$` . This basic measure is not useful.

Another way is to use modified precision in which we are looking for the reference with the maximum number of a particular word and set the maximum appearing of this word to this number. So, `$modifiedPrecision = \frac{2}{7}$` because the max is 2 in Y1. We clipped the 7 times by the max which is 2.

Here we are looking at one word at a time, we may need to look at pairs.

*Bleu Score on bigrams:*

Suppose our predicted statement is :

- `$\hat{y}$` : "the cat the cat on the mat."

The output in here:

Pairs | Count | Count clip
:-: | :-: | :-:
the cat	| 2	| 1 (Y1)
cat the	| 1	| 0
cat on	| 1	| 1 (Y2)
on the	| 1	| 1 (Y1)
the mat	| 1	| 1 (Y1)
Totals	| 6	| 4

Score : Count clip / count : 4 / 6

For n-grams, Score will be:

`$$ P_n = \frac{\sum_{n-gram\in\hat{y}}(CountClip_{n-gram})}{\sum_{n-gram\in\hat{y}}(Count_{n-gram})} $$`

*Bleu Score Combined:*

- `$p_n$` : Bleu score on n-grams.
- Suppose we calculate for n=1,2,3,4. Then:

`$$ score = BP*exp(\frac{1}{4}\sum_{n=1}^4p_n) $$`

BP: penalty which stands for brevity penalty. Penalizes the sentences shorter than the target.

![](https://github.com/myselfHimanshu/data-summit-blog/raw/master/images/part3_5.png)

**Attention Model**

The problem of long sequence:

![](https://github.com/myselfHimanshu/data-summit-blog/raw/master/images/part3_6.png)

- In encoder and decoder network, first we are trying to get the representation of the long french statement and then ask decoder to process and generate the translated sentence.
- But in real world, a human wouldn't wait for the end of the sentence and then memorize it then try to translate it. He will translate a little at a time, part by part.
- Encoder and decoder network score comes down for longer sentences.
- Attention model works like a human that looks at parts at a time. This significantly increases the accuracy with bigger sequences.

![](https://github.com/myselfHimanshu/data-summit-blog/raw/master/images/part3_7.png)

- Our network is a bidirectional RNN for our input `X`. Let the activations be

`$$ a^{<t^{'}>} = f(\overrightarrow{a}^{<t>}, \overleftarrow{a}^{<t>} ) $$`

- Then we gonna use another RNN network for our translated output.
- Suppose the first word that we want to generate is `Jane`. To generate `Jane`, we don't have to look for the whole sentence, we just need to look at a part of a sentence (maybe some few first words).
- The attention model computes something called as `attention-weights` (`$\alpha^{<1,2>}$$, $$\alpha^{<1,2>}$`...).
- `attention-weights` : if we are generating the first output word, how much attention should we pay to the first input word and second input word and so on.

`$$\sum_{t^{'}}\alpha^{<t,t^{'}>} = 1$$`

- Together `attention-weights` will tell us which all contexts in `C` we should pay attention to.

`$$c^{<t>} = \sum_{t^{'}}\alpha^{<t,t^{'}>}a^{<t^{'}>} $$`


- For each state `$S^{<i>}$` we will have a new set of `attention-weights`.
- In other words, to generate any word there will be a set of attention weights that controls which words we are looking at right now.

*How to compute these attention-weights?*

We softmax the attention weights so that their sum is 1

`$$\alpha^{<t,t^{'}>} = \frac{exp(e^{<t,t^{'}>})}{\sum_{t^{'}=1}^{T_x}exp(e^{<t,t^{'}>})}$$`

We calculate `$e^{<t,t^{'}>}$` using a small neural network:
![](https://github.com/myselfHimanshu/data-summit-blog/raw/master/images/part3_8.png)

- `$s^{<t-1>}$` is the hidden state of the generator RNN from previous time step, and $$a^{<t'>}$$ is the activation of the our input bidirectional RNN.
- The disadvantages of this algorithm is that it takes quadratic time or quadratic cost to run.

## Problem Statements

*Coursera Assignment*

- <a href="https://github.com/myselfHimanshu/Coursera-DataML/blob/master/deeplearning.ai/Course5/Neural%2Bmachine%2Btranslation%2Bwith%2Battention%2B-%2Bv4.ipynb" target="_blank">Neural Machine Translation with Attention.</a>
- <a href="" target="_blank">Trigger word detection.</a>
