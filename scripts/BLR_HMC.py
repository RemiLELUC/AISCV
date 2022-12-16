#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author: Rémi LELUC, François PORTIER, Johan SEGERS, Aigerim ZHUMAN
This file contains the functions to perform Hamiltonian Monte Carlo.
'''

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.datasets import load_svmlight_file
import pymc3 as pm
from pymc3 import Model, sample

def load_data(filename):
    S,y = load_svmlight_file(f=filename)
    X = csr_matrix.toarray(S)
    return X,y

def g(v):
    return (v.T).dot(v)

## Load dataset ## uncomment line to load different datasets
#df = pd.read_csv('./datasets/winequality-red.csv',sep=';')
#df = pd.read_csv('./datasets/winequality-white.csv',sep=';')
# data = np.array(df)
#X = data[:,:-1]
#y = data[:,-1]

#X,y = load_data('./datasets/abalone')  
X,y = load_data('./datasets/housing')  
#print(X.shape)
n_samples,n_features = X.shape

# Construct NUTS Model
basic_model = Model()
with basic_model:
    # parameter of interest
    theta = pm.Normal('theta', mu=0, sigma=50, shape=n_features)
    # Expected value of outcome for BLR
    mu = X@theta
    # Likelihood (sampling distribution) of observations
    Y_obs = pm.Normal('Y_obs', mu=mu, sigma=50, observed=y)


n_chains = 30      # Number of chains to run
n_draws = 3000     # Number of draws for each chain
n_cores = 40       # Number of cores for parallel sampling
seed = 42          # Random seed for reproducibility
tune_samples = 400 # Number of samples for Tuning of HMC

with basic_model:
    # draw 1000 posterior samples
    trace = pm.sample(draws=n_draws,init='adapt_diag',compute_convergence_checks=False,
                      tune=tune_samples,random_seed=seed,cores=n_cores, chains=n_chains, return_inferencedata=False)

vals = trace.get_values('theta')
eval_cut = np.array(np.vsplit(vals,n_chains))

eval_g = np.zeros((n_chains,3000))
for i in range(n_chains):
    eval_g[i] = np.array([g(val) for val in eval_cut[i]])

hmc_val = np.zeros((5,n_chains))
hmc_val[0] = np.mean(eval_g[:,:5000],axis=1)
hmc_val[1] = np.mean(eval_g[:,:10000],axis=1)
hmc_val[2] = np.mean(eval_g[:,:20000],axis=1)
hmc_val[3] = np.mean(eval_g[:,:30000],axis=1)

#np.save('hmc_val_housing.npy',hmc_val)
#np.save('hmc_val_abalone.npy',hmc_val)
#np.save('hmc_val_redwine.npy',hmc_val)
#np.save('hmc_val_whitewine.npy',hmc_val)
