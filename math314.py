import numpy as np
import multiprocessing
from scipy.optimize import minimize
from joblib import Parallel, delayed

def ll_normal(beta, yX):
    y = yX[:, 0]
    X = yX[:, 1:]
    N = X.shape[0]
    mu = np.full(N, np.nan)
    for n in range(N):
        mu[n] = np.sum(X[n, :] * beta)
        
    d = y - mu
    return np.sum(d*d)

def bootstrap(data, R, fun, initval = None):
    N=data.shape[0]
    k=data.shape[1]
    k -= 1
    thetas = np.full((R, k), np.nan)
    for r in range(R):
        idx = np.random.choice(N, N, replace=True)
        thetas[r, :] = fun(data[idx, :], initval)
    return thetas
        
def optim(data, initval = None):
    k = data.shape[1]-1
    if not np.any(initval):
        initval = np.random.normal(size = k)
    return minimize(ll_normal, initval, args=(data), method="BFGS")["x"]

def boot(data, N, fun, initval):
    idx = np.random.choice(N, N)
    return fun(data[idx, :], initval)

def pbootstrap(data, R, fun, initval = None, ncpus=4):
    N=data.shape[0]
    thetas = Parallel(ncpus)(delayed(boot)(data, N, fun, initval) for _ in range(R))
    return np.asarray(thetas)

def adjustedR2(y, mu, k):
    error = y - mu
    N= y.shape[0]
    return 1 - (np.var(error) / np.var(y)) * (N-1)/(N-k)