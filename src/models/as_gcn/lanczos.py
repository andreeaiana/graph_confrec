import numpy as np
from numpy.linalg import norm
import scipy.sparse as sparse
import pickle
from scipy.linalg import qr, svd

# DISCLAIMER:
# This code file is forked from https://github.com/huangwb/AS-GCN


def lanczos(A, k, q):
    n = A.shape[0]
    Q = np.zeros((n, k+1))

    Q[:, 0] = q/norm(q)

    alpha = 0
    beta = 0

    for i in range(k):
        if i == 0:
            q = np.dot(A, Q[:, i])
        else:
            q = np.dot(A, Q[:, i]) - beta * Q[:, i-1]
        alpha = np.dot(q.T, Q[:, i])
        q = q - Q[:, i] * alpha
        # full reorthogonalization
        q = q - np.dot(Q[:, :i], np.dot(Q[:, :i].T, q))
        beta = norm(q)
        Q[:, i+1] = q/beta
        print(i)
    Q = Q[:, :k]
    Sigma = np.dot(Q.T, np.dot(A, Q))

    return Q, Sigma


def dense_RandomSVD(A, K):
    G = np.random.randn(A.shape[0], K)
    B = np.dot(A, G)
    Q, R = qr(B, mode='economic')
    M = np.dot(np.dot(Q, np.dot(np.dot(Q.T, A), Q)), Q.T)
    return M
