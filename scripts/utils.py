#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author: Rémi LELUC, François PORTIER, Johan SEGERS, Aigerim ZHUMAN
This file contains tool functions for simulations.
'''


import numpy as np
from scipy.stats import multivariate_normal
from scipy.special import gammaln


def multivariate_uniform_pdf(x):
    ''' Probability density of Uniform[0,1] distribution '''
    return np.logical_and(x>0,x<1).all(axis=1).astype(int)


def multivariate_student_generate(m, S, df=np.inf, n=1):
    '''generate random variables of multivariate t distribution'''
    m = np.asarray(m)
    dim = len(m)
    if df == np.inf:
        x = 1.
    else:
        x = np.random.chisquare(df, n)/df
    z = np.random.multivariate_normal(np.zeros(dim),S,(n,))
    return m + z/np.sqrt(x)[:,None]   


def multivariate_student_pdf(x, mean, shape, df):
    ''' Probability density of multivariate t distribution '''
    return np.exp(logpdf(x, mean, shape, df))


def logpdf(x, mean, shape, df):
    ''' Log-Probability density of multivariate t distribution '''
    dim = mean.size

    vals, vecs = np.linalg.eigh(shape)
    logdet     = np.log(vals).sum()
    valsinv    = np.array([1./v for v in vals])
    U          = vecs * np.sqrt(valsinv)
    dev        = x - mean
    maha       = np.square(np.dot(dev, U)).sum(axis=-1)

    t = 0.5 * (df + dim)
    A = gammaln(t)
    B = gammaln(0.5 * df)
    C = dim/2. * np.log(df * np.pi)
    D = 0.5 * logdet
    E = -t * np.log(1 + (1./df) * maha)

    return A - B - C - D + E


def get_combinations(n, k):
    """  gets all lists of size k summing to exactly n """
    ans = []
    def solve(rem, depth, k, cur):
        if depth == k:
            ans.append(cur)
        elif depth == k-1:
            solve(0, depth+1, k, cur + [rem])
        else:
            for i in range(rem+1):
                solve(rem-i, depth+1, k, cur+[i])
    solve(n, 0, k, [])
    return ans


def get_comb_lower(Q,size):
    """  gets all lists of size k summing to lower or equal to n """
    res = []
    for i in range(1,Q+1):
        res += get_combinations(n=i,k=size)
    return np.array(res)
