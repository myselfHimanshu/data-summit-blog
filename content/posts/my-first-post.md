+++
draft = false
date = 2019-11-05T19:20:13+05:30
title = "My First Blog Post : Algorithm"
slug = ""
tags = ["ALGORITHMS"]
categories = []
math = "true"
+++

Here's a small python code.

### Sum of two digits

Compute the sum of two single digit numbers

**Input format** : Integers a and b on the same line (separated by a space).<br />
**Output format** : The sum of a and b.<br />
**Constraints** : `$0<=a,b<=9$`


``` python
#Uses python3
import sys

input = sys.stdin.read()
tokens = input.split()
a = int(tokens[0])
b = int(tokens[1])
print(a+b)
```

### Maximum Pairwise Product

Find the maximum product of two distinct numbers in a sequence of non-negative integers.<br />

**Input format** : The first line contains an integer. The next line contains n non-negative integers `$a_1.....a_n$` (separated by spaces).<br />
**Output format** : The maximum pairwise product.<br />
**Constraints** : `$2<=n<=2.10^5; 0<=a_1....a_n<=2.10^5$`


``` python
#uses python3
n = int(input())
a = [int(x) for x in input().split()]

#print(a)

result = 0
max1 = 0
max2 = 0
index1 = 0
index2 = 0

for i in range(n):
	if a[i]>max1:
		max1 = a[i]
		index1 = i

for i in range(n):
	if a[i]>max2 and index1!=i:
		max2 = a[i]
		index2 = i

#print(max1, max2)
print(max1 * max2)

```
