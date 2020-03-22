+++
draft = true
date = 2020-02-17T11:15:02+05:30
title = "Statistics and Probability"
slug = ""
tags = ["STATS","DATA"]
categories = []
math = "true"
+++

## Concepts

**Identifying individuals, variables and categorical variables in data**

Name | Age | Gender | Survived
:--:|:-:|:-:|:-:|
Ab|27|Male|Yes
Em|24|Female|Yes
Pm|48|Female|Yes
Ro|99|Male|No
Lm|72|Female|No

Here if we are asked which ones are individuals and how many variables are there and which variables are categorical.

> **Answer** : `Name` is individuals, it's not a variable, it's more like an identifier. `Age`, `Gender`, 'Survived' are variables and `Gender`,`Survived` are categorical variables.



**Statistical Features**

This is used at the time of exploring the dataset. It contains information like bias, variance, mean, median, percentiles etc.
The best way to get these information is to plot a boxplot. It's a standard way of displaying the distribution of data.

![](https://github.com/myselfHimanshu/Portfolio-Website/blob/master/images/boxplot.png?raw=true)

If boxplot is small then it means that many values are same or in a small range, if it is tall that means out data points are different. If the median value is closer to bottom that means most of the data have lower values which means if the median value is not around middle of the box then we have skewed data.  

**Probability Distribution**

A function which helps you estimate the probability of any measurement within your population being within a specific range.

If a random variable is discrete, its probability distribution is called a discrete probability distribution.

Variance is one way to measure the spread in a data set, and it is defined as the sum of the squared deviations from the mean. For a discrete probability distribution, variance can be calculated using the following equation:

$$ Var(x) = \sum{p_i (x_i)^2} - [E(x)]^2$$

Standard Deviation is simply equal to the square root of the variance:
$$ \sigma = \sqrt{Var(x)} $$

Types of discrete probability distributions:

1. Bernoulli Distribution :
  - It is the simplest kind of discrete probability distribution that can only take two possible values, usually 0 and 1 (success and failure).<br>
  - It is important that the variable being measured is both random and independent.<br>
  - Variance for Bernoulli distribution : $$ Var = P(1-P) $$
2. Poisson Distribution :
  - It is used to calculate the probability of an event occurring over a certain interval.<br>
  - The interval can be one of time, area, volume or distance. Usually, it can be applied to systems with a large number of possible events, each of which is rare.<br>

$$ \frac{\lambda^k e^{-\lambda}}{k!} $$

where,<br>
k = 0,1,2,3…(number of successes for the event)<br>
e = 2.71828 (Euler’s constant)<br>
λ = mean number of successes in the given time interval or region of space<br>

Two conditions of Poisson distribution<br>
1. Each successful event must be independent. <br>
2. The probability of success over a short interval must equal the probability of success over a longer interval. <br>



**Bayesian Statistics**

|Frequentist Statistics|Bayesian Statistics|
|:-:|:-:|
Parameters Fixed | Parameters Vary
Data Varies | Data Fixed
Probability P(D/O) | Likelihood P(D/O)
Confidence Interval | Credible Interval
No Prior | Strength of Prior

Bayes Theorem

$$ P(A|B) = \frac{P(B|A)P(A)}{P(B)} $$

P(A|B) : The probability of 'A' being True given that 'B' is true.<br>
P(B|A) : The probability of 'B' being True given that 'A' is true.<br>
P(A) : The probability of 'A' being True.<br>
P(B) : The probability of 'B' being True.<br>

**Random Variables**

Random Variable : whose possible values are outcomes of a random phenomenon.

![](https://github.com/myselfHimanshu/Portfolio-Website/blob/master/images/continuous-variable.jpg?raw=true)

*Uniform Distribution*<br>
A uniform distribution, sometimes also known as a rectangular distribution, is a distribution that has constant probability.

*Normal Distribution*<br>
Data can be "distributed" (spread out) in different ways. But there are many cases where the data tends to be around a central value with no bias left or right, and it gets close to a "Normal Distribution" like this:

![](https://github.com/myselfHimanshu/Portfolio-Website/blob/master/images/bell-curve.jpg?raw=true)

**Skewness**<br>
It is the degree of distortion from the symmetrical bell curve or the normal distribution.

The rule of thumb seems to be:

- If the skewness is between -0.5 and 0.5, the data are fairly symmetrical.
- If the skewness is between -1 and -0.5(negatively skewed) or between 0.5 and 1(positively skewed), the data are moderately skewed.
- If the skewness is less than -1(negatively skewed) or greater than 1(positively skewed), the data are highly skewed.

**Kurtosis**<br>
Kurtosis is all about the tails of the distribution — not the peakedness or flatness. It is used to describe the extreme values in one versus the other tail. It is actually the measure of outliers present in the distribution.


**Central Limit Theorem**<br>
The central limit theorem (CLT) is one of the most important results in probability theory. It states that, under certain conditions, the sum of a large number of random variables is approximately normal.

The gist is we can take some random samples (randomly selected and samples are big enough) from the large population then the mean of samples will be closely approximate to the mean of overall population. This is basically concerned with sampling distribution of the mean, in other words, repeatedly take random samples of size n, calculate the mean of each sample, analyze the distribution.

Then we look at the distribution of those means and come to 4 conclusions:

$$\mu = populationMean , \sigma = population-std-dev, n = sample size $$

1. The sampling distribution of the mean will be less spread than the values in the population from which the samples are drawn.
2. The sampling distribution will be well modeled by a normal distribution.
3. The spread of the sampling distribution is related to the spread of the population values(mean = $\mu$, std_div = $\frac{\sigma}{\sqrt{n}}$)
4. Bigger samples lead to a smaller spread in the sampling distribution.

Here's the link for <a href="https://gist.github.com/myselfHimanshu/d2c638191909abb7e16581a2fbc945b7" target="_blank">NYCflights13 data analysis</a>.
