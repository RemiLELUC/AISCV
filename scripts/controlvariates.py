#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author: Rémi LELUC, François PORTIER, Johan SEGERS, Aigerim ZHUMAN
This file contains functions to get matrix H = (h_j(X_i)) of size n x k x d
of the control variates, in Legendre/Hermite/Laguerre/Fourier basis.
It also implements a simple function to draw random samples
'''

# Import libraries
import numpy as np
# Polynomial families for control variates
from numpy.polynomial.hermite_e import hermeval
from numpy.polynomial.legendre import legval
from numpy.polynomial.laguerre import lagval
from scipy.special import factorial

def draw_sample(n,d,law='normal'):
    ''' Function to draw n samples in dimension d of some distribution
    Params
    @n (int)  : number of samples to draw
    @d (int)  : dimension of the problem
    @law (str): distribution 'uniform' or 'normal'
    Returns
    @X (array n x d): random samples X_1,...,X_n
    '''
    if law == 'uniform': # Uniform law on [0,1]^d
        X = np.random.rand(n, d)
    elif law == 'normal': # Normal law on R^d
        X = np.random.randn(n, d)
    return X

def primes_from_2_to(n):
    """Prime number from 2 to n.
    From `StackOverflow <https://stackoverflow.com/questions/2068372>`_.
    :param int n: sup bound with ``n >= 6``.
    :return: primes in 2 <= p < n.
    :rtype: list
    """
    sieve = np.ones(n // 3 + (n % 6 == 2), dtype=np.bool)
    for i in range(1, int(n ** 0.5) // 3 + 1):
        if sieve[i]:
            k = 3 * i + 1 | 1
            sieve[k * k // 3::2 * k] = False
            sieve[k * (k - 2 * (i & 1) + 4) // 3::2 * k] = False
    return np.r_[2, 3, ((3 * np.nonzero(sieve)[0][1:] + 1) | 1)]

def van_der_corput(n_sample, base=2):
    """Van der Corput sequence.
    :param int n_sample: number of element of the sequence.
    :param int base: base of the sequence.
    :return: sequence of Van der Corput.
    :rtype: list (n_samples,)
    """
    sequence = []
    for i in range(n_sample):
        n_th_number, denom = 0., 1.
        while i > 0:
            i, remainder = divmod(i, base)
            denom *= base
            n_th_number += remainder / denom
        sequence.append(n_th_number)

    return sequence

def halton(dim, n_sample):
    """Halton sequence.
    :param int dim: dimension
    :param int n_sample: number of samples.
    :return: sequence of Halton.
    :rtype: array_like (n_samples, n_features)
    """
    big_number = 10
    while 'Not enought primes':
        base = primes_from_2_to(big_number)[:dim]
        if len(base) == dim:
            break
        big_number += 1000

    # Generate a sample using a Van der Corput sequence per dimension.
    sample = [van_der_corput(n_sample + 1, dim) for dim in base]
    sample = np.stack(sample, axis=-1)[1:]

    return sample
############### CONTROL VARIATES FAMILIES ###############
def legendre(k,t):
    ''' Legendre Polynomial of degree k at point t 
    Params
    @k   (int): order of Legendre polynomial function
    @t (float): point to be evaluated
    Returns: (float) Leg_k(t)
    '''   
    c = np.zeros(k+1)
    c[-1] = 1
    return 2*legval(2*t-1, c)

def hermite(k,t):
    ''' Hermite Polynomial of degree k at point t 
    Params
    @k   (int): order of Hermite polynomial function
    @t (float): point to be evaluated
    Returns: (float) Her_k(t)
    '''
    c = np.zeros(k+1)
    c[-1] = 1
    return hermeval(t, c)

def hermite_exp(k,t):
    ''' Hermite Polynomial of degree k at point t 
    Params
    @k   (int): order of Hermite polynomial function
    @t (float): point to be evaluated
    Returns: (float) Her_k(t)
    '''
    c = np.zeros(k+1)
    c[-1] = 1
    return hermeval(t, c) * np.exp(-t**2/2)

def hermite_norm(k,t):
    ''' Hermite Polynomial of degree k at point t 
    Params
    @k   (int): order of Hermite polynomial function
    @t (float): point to be evaluated
    Returns: (float) Her_k(t)
    '''
    c = np.zeros(k+1)
    c[-1] = 1
    return hermeval(t, c)/factorial(k)

def laguerre(k,t):
    ''' Laguerre Polynomial of degree k at point t 
    Params
    @k   (int): order of Laguerre polynomial function
    @t (float): point to be evaluated
    Returns: (float) Lag_k(t)
    '''
    c = np.zeros(k+1)
    c[-1] = 1
    return lagval(t, c)

def fourier(k,t):
    ''' Fourier basis function of order k at point t
    Params
    @k   (int): order of basis fourier function
    @t (float): point to be evaluated
    Returns: (float) Fourier_k(t)
    '''
    if k%2==1:
        return np.sqrt(2)*np.cos((k+1)*np.pi*t)
    else:
        return np.sqrt(2)*np.sin(k*np.pi*t)
    
############### Compute Matrix H of size n x k x d ############### 
def get_H_nkd(X,k,basis='legendre'):
    ''' Compute Tensor matrix H = (h_j(X_i))of size n x k x d in the given basis
    Params
    @X (array n x d): random samples X_1,...,X_n
    @k (int)        : number of control variates in each direction
    @basis (str)    : name of the basis in ['legendre','hermite','laguerre','fourier']
    Returns
    @H : matrix H = (h_j(X_i)) of size n x k x d in the given basis
    '''
    n,d = X.shape
    H = np.zeros((n,k,d))
    # Fill the matrix H in the given basis
    if basis=='legendre':
        for j in range(1,k+1):
            H[:,j-1,:] = legendre(j,X)
    elif basis=='hermite':
        for j in range(1,k+1):
            H[:,j-1,:] = hermite(j,X)
    elif basis=='hermite_norm':
        for j in range(1,k+1):
            H[:,j-1,:] = hermite_norm(j,X)
    elif basis=='hermite_exp':
        for j in range(1,k+1):
            H[:,j-1,:] = hermite_exp(j,X)
    elif basis=='laguerre':
        for j in range(1,k+1):
            H[:,j-1,:] = laguerre(j,X)
    elif basis=='fourier':
        for j in range(1,k+1):
            H[:,j-1,:] = fourier(j,X)
    return H
####################################################################

def mask_single(k,d):
    I = np.eye(d)
    mask = np.eye(d)
    for i in range(2,k+1):
        mask = np.concatenate((mask,i*I))
    return mask.astype(int)

def mask_pairs(k,d):
    tab_list = []
    for i in range(d-1):
        for j in range(i+1,d):
            for i_k in range(0,k+1):
                for j_k in range(0,k+1):
                    tab = [0]*d
                    tab[i] = i_k
                    tab[j] = j_k
                    tab_list.append(tab)
    mask = np.delete(arr=sorted(list(set(tuple(row) for row in tab_list)),key=sum),obj=0,axis=0)
    return mask.astype(int)
