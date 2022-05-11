import numpy as np
from sklearn.decomposition import PCA, TruncatedSVD
import pandas as pd
from scipy.linalg import hadamard
import numpy as np
from numpy.linalg import norm
from cmath import sqrt

def optimal_probabilities(A, B):
    '''
    Proposed optimal initial probability as on Drines, Kannan & Mahoney's first paper (p. 141)
    IN    - A : m x n real matrix
          - B : n x q real matrix
    OUT   - p : 1xn vector with initial optimal probability distribution    
    '''
    n = len(B)
    p = [0]*n
    normalizing_sum = 0
    for i in range(n):
        normalizing_sum = normalizing_sum + norm(A[:, i])* norm(B[i, :]) 
        p[i] = norm(A[:, i])* norm(B[i, :]) 
    p = p/normalizing_sum
    return p

m = 2**5
# Ua = hadamard(m)
# Va = hadamard(2*m)
# s = np.zeros([m,1])
# for k in range(m-1):
#     s[k] = 2/(k+1)
# S = np.zeros([m, 2*m])
# np.fill_diagonal(S, s)
# A = Ua@S@np.transpose(Va)


# U, S, Vt = np.linalg.svd(A)
# k = 3
# S_aux = np.zeros([k, k])
# print( np.diag(S[0:k]) )

print(list(range(3)))