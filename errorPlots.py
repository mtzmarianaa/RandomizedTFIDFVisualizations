# FUNCTIONS PRESENTED IN DRINEAS, KANNAN & MAHONEY PAPERS
from cmath import sqrt
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm
from random import random 

def basicMatrixMultiplications(A, B, c, p):
    '''
    Algorithm proposed on Drineas, Kannan & Mahoney's first paper (out of 3), p. 138
    IN    - A : m x n real matrix
          - B : n x p real matrix
          - c : real number such that 1<= c <= n
          - p : 1xn vector with initial probability distribution
    OUT   - C : m x c real matrix
          - R : c x p real matrix such that AB approx CR
    '''
    m, n = A.shape()
    possible_indices = range( n )
    C = np.zeros((m, c))
    R = np.zeros((n, c))
    for t in range(c):
        it = np.random.choice( possible_indices, p  )
        C[:, it] = A[:, it]/sqrt(c*p[it])
        R[it, :] = B[it, :]/sqrt(c*p[it])
    return C, R
        
        




