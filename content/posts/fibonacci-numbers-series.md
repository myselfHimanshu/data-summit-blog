+++
draft = false
date = 2019-11-06T12:31:51+05:30
title = "Fibonacci Numbers Series"
slug = ""
tags = ["ALGORITHMS"]
categories = []
math = "true"
+++

### Fibonacci Number

Fibonacci sequence: `$F_0=0$`, `$F_1=1$`, and `$F_i=F_{i-1}+F_{i-2}$`. <br />
Given an integer n, find the nth Fibonacci number $F_n$

**Input format** : The input consists of a single integer n.<br />
**Output format** : Output $F_n$.<br />
**Constraints** : `$0<=n<=45$`

**CODE**
```python3
a = [0,1]
for i in range(2,46):
 	a.append(a[i-1] + a[i-2])

def calc_fib(n):
    return a[n]

n = int(input())
print(calc_fib(n))
```

### Last Digit of Large Fibonacci Number

Given an integer n, find the last digit of the nth Fibonacci number $F_n$(that is, $F_n$ mod 10)

The previous solution will turn out to be slow, because as i grows the ith iteration of the loop computes the sum of longer and longer numbers.

**Input format** : The input consists of a single integer n.<br />
**Output format** : Output the last digit of $F_n$.<br />
**Constraints** : `$0<=n<=10^7$`

**CODE**
```python3
a = [0,1]
for i in range(2,10**7+1):
 	a.append((a[i-1] + a[i-2])%10)

def calc_fib(n):
    return a[n]

n = int(input())
print(calc_fib(n))
```

### Compute Huge Fibonacci Number modulo m

Given two integers n and m, output $F_n$ mod m (that is, the remainder of $F_n$ when divided by m).

**Input format** : The input consists of two integers n and m given on the same line (separated by a space).<br />
**Output format** : Output $F_n$ mod m.<br />
**Constraints** : `$1<=n<=10^{18}$`,`$2<=m<=10^{5}$`

For such values of n, an algorithm looping for i iterations will not fit into one second for sure. Let's compute for small m.

i|0|1|2|3|4|5|6|7|8|9|10|11|12|13|14|15
:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:
$F_i$|0|1|1|2|3|5|8|13|21|34|55|89|144|233|377|610
$F_i mod 2$|0|1|1|0|1|1|0|1|1|0|1|1|0|1|1|0
$F_i mod 3$|0|1|1|2|0|2|2|1|0|1|1|2|0|2|2|1

Both the sequences are periodic. For m=2, the period is 001 and for m=3, the period is 01120221.

This is true in general: for any integer m≥2, the sequence $F_n$ mod m is periodic. The period always starts with `01` and is known as `Pisano period`.

**CODE**
```python3
import sys

def get_fibonacci_huge(n, m):
    if n <= 1:
        return n

    previous = 0
    current  = 1
    fib_result = [0,1]

    for i in range(n-1):
        previous, current = current%m, (previous + current)%m
        fib_result.append(current)
        if previous==0 and current==1:
            return fib_result[n%(i+1)]

    return current

if __name__ == '__main__':
    input_ = sys.stdin.readline()
    n, m = map(int, input_.split())
    print(get_fibonacci_huge(n, m))
```

### Last Digit of Sum of Fibonacci Number

Given an integer n, find the last digit of the sum `$F_0+F_1+.....+F_n$`.

**Input format** : The input consists of single integer n.<br />
**Output format** : Output the last digit of `$F_0+F_1+.....+F_n$`.<br />
**Constraints** : `$1<=n<=10^{14}$`

**CODE**
```python3
import sys

def get_fibonacci_huge(n, m):
    if n <= 1:
        return n

    previous = 0
    current  = 1
    fib_result = [0,1]

    for i in range(n-1):
        previous, current = current%m, (previous + current)%m
        if previous==0 and current==1:
            fib_result.pop()
            return fib_result
        fib_result.append(current)    
    return fib_result

def fibonacci_sum(n):
    if n <= 1:
        return n

    sum_ = 0
    array = get_fibonacci_huge(n, 10)
    q,r = n//len(array), n%len(array)
    sum_ = (sum_+((sum(array)%10)*q))%10
    sum_ = sum_ + sum(array[:r+1])%10
    return sum_ % 10

if __name__ == '__main__':
    input_ = sys.stdin.readline()
    n = int(input_)
    print(fibonacci_sum(n))
```

### Last Digit of Sum of Fibonacci Number Again

Given two non-negative integers m and n, where $m<=n$ find the last digit of the sum `$F_m+F_{m+1}+.....+F_n$`.

**Input format** : The input consists of two non-negative integers m and n separated by a space.<br />
**Output format** : Output the last digit of `$F_m+F_{m+1}+.....+F_n$`.<br />
**Constraints** : `$0<=m<=n<=10^{18}$`

**CODE**
```python3
import sys

def get_fibonacci_huge(n , m):
    if n <= 1:
        return n

    previous = 0
    current  = 1
    fib_result = [0,1]

    for i in range(n-1):
        previous, current = current%m, (previous + current)%m
        if previous==0 and current==1:
            fib_result.pop()
            return fib_result
        fib_result.append(current)

    return fib_result

def fibonacci_partial_sum(k, n):
    if n <= 1:
        return n

    sum_  = 0

    array = get_fibonacci_huge(n, 10)
    q,r = n//len(array), n%len(array)
    q1,r1 = k//len(array), k%len(array)

    sum_ = (sum_+(sum(array[r1:])%10))%10
    sum_ = (sum_+((sum(array)%10)*(q-q1-1)))%10
    sum_ = (sum_+(sum(array[:r+1])%10))%10

    return sum_ % 10

if __name__ == '__main__':
    input_ = sys.stdin.readline()
    from_, to = map(int, input_.split())
    print(fibonacci_partial_sum(from_, to))
```
