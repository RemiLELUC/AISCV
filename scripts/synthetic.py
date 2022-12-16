#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author: Rémi LELUC, François PORTIER, Johan SEGERS, Aigerim ZHUMAN
This file contains functions to perform experiments with synthetic integrands.
'''


# Import libraries
import numpy as np
from model import AISCV


###########################
### Integrands g1,g2,g3 ###
###########################

# Sinusoidal function
def phi(x):
    return 1 + np.sin(np.pi*(2*np.mean(x) - 1))
lbda = np.log(2)
# Log-normal density function with mean=0 and variance=1
def f(x):
    return np.prod(2*lognorm.pdf(x=x,s=1),axis=0)
# Exponential density function with parameter lambda = ln(2)
def h(x):
    return np.prod(2*expon.pdf(x=x,scale=1/lbda),axis=0)

#######################
### Hyperparameters ###
#######################

basis='legendre' # basis of control variates
mode = 'pairs'   # tensor combinations for control variates
n_t = 1000       # allocation policy (fixed)
sigma_0 = 0.1    # variance of covariance student matrix 
df = 8           # degrees of freedom of student distribution
k = 6            # number of control variates in each direction
N_exp = 100      # total number of replications
I_true = 1       # true value of integral
list_f = [phi,f,h]
list_names = ['sin','log_norm','expo']

### Run experiments
for j in range(3):
    name = list_names[j]
    current_f = list_f[j]
    # loop over dimensions
    for d in [4,8]:
        print('pairs: ',int(k*d + k*k*d*(d-1)/2))
        I_ais = np.zeros((5,N_exp))
        I_wais = np.zeros((5,N_exp))
        I_aiscv = np.zeros((5,N_exp))
        for i,T in enumerate([5,10,20,30,50]):
            # instance of MC estimate
            mc = AISCV(func=current_f)
            mc.run_multi(N_exp=N_exp,d=d,T=T,n_t=n_t,sigma_0=sigma_0,
                         df=df,basis=basis,mode=mode,k=k)
            I_ais[i] =  mc.res_ais
            I_wais[i] =  mc.res_wais
            I_aiscv[i] = mc.res_aiscv
        np.save('res_ais_d{}_Nexp{}_'.format(str(d),str(N_exp))+name+'.npy',I_ais)
        np.save('res_wais_d{}_Nexp{}_'.format(str(d),str(N_exp))+name+'.npy',I_wais)
        np.save('res_aiscv_d{}_Nexp{}_'.format(str(d),str(N_exp))+name+'.npy',I_aiscv)
