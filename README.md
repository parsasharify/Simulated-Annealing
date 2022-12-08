# Simulated-Annealing
In This Problem, We want to investigate the subset sum problem. Informally, find a subset from a given set of numbers that their sum is equal to a given number. For example, if the given set is $ {1, 2, 3, 4, 5}$  and the given number is $ 10 $, then the subset $ {1, 2, 3, 4} $ is a solution. One important assumption that we make is that the given set is a set of positive integers. In this problem, we want to find a solution for the subset sum problem using Simulated Annealing.
The Formal definition of the problem is as follows:
Given a set of positive integers $ S $ and a positive integer $ k $, find a subset $ S' $ of $ S $ such that $ \sum_{i \in S'} i = k $.

We call an answer feasible if it is a subset of $ S $ and its sum is **less than or equal** to $ k $. (i.e. $ \sum_{i \in S'} i \leq k $)
This variant of Subset Sum is a famous NP-Complete optimization problem. It means that we currently don't have any polynomial-time algorithm for this problem. Therefore it is reasonable to use optimization algorithms like local search to find an approximate but not necessarily perfect answer.
