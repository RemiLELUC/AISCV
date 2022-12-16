#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author: Rémi LELUC, François PORTIER, Johan SEGERS, Aigerim ZHUMAN
This file contains functions to perform experiments with Gaussian mixtures
'''


# Basic library for manipulating arrays
import numpy as np
from utils import multivariate_student_pdf, multivariate_student_generate
from utils import get_comb_lower
from tqdm import tqdm
# Library to perform OLS
from sklearn.linear_model import LinearRegression
# Library to compute gradient automatically
import torch
from torch import tensor, eye, sqrt, zeros, log, exp, ones
from torch.distributions import MultivariateNormal as MVN

print('torch version: ',torch.__version__)

# integrand for mean a posteriori
def g(𝜃):
    return 𝜃

def target(𝜃):
    """ isotropic gaussian mixtures """
    p = tensor(𝜃.shape[1])
    𝜃 = tensor(𝜃,requires_grad=False).double()
    shift = ones(p)/(2*sqrt(p))
    μs   = zeros(p)               # mean
    Σs   = (1/p)*eye(p)           # covariance
    MVN_U = MVN(μs,Σs)            # Multivariate Normal 
    eval_f = 0.5 * exp(MVN_U.log_prob(𝜃-shift).double()) + 0.5 * exp(MVN_U.log_prob(𝜃+shift).double())
    return eval_f.detach().numpy()
    
def grad_log_f(𝜃):
    """ score function for isotropic gaussian mixtures """
    p = tensor(𝜃.shape[1])
    𝜃 = tensor(𝜃,requires_grad=True).double()
    shift = ones(p)/(2*sqrt(p))
    μs   = zeros(p)             # mean
    Σs   = (1/p)*eye(p)         # covariance
    MVN_U = MVN(μs,Σs)          # Multivariate Normal

    log_f = log( 0.5 * exp(MVN_U.log_prob(𝜃-shift).double()) + 0.5 * exp(MVN_U.log_prob(𝜃+shift).double()) )
    # compute gradient
    log_f.backward(torch.ones(𝜃.shape[0]))
    return (𝜃.grad).detach().numpy()

def run(func,seed,d,Q,T,n_t,df):
    """ Run 1 experiment of AIS and AISCV on isotropic gaussian mixtures
    Params:
    @func (function): integrand g of the problem
    @seed      (int): random seed 
    @d         (int): dimension of the problem
    @Q         (int): total bounded degree for control variates
    @T         (int): number of stages
    @n_t       (int): number of samples at each stage
    @df        (int): degrees of freedom for multivariate student law
    Returns:
    I_ais,I_wais,I_aiscv
    """
    # Create all permutations of powers alphas
    mask = get_comb_lower(Q=Q,size=d)
    m = len(mask)
    # set random seed for the run
    np.random.seed(seed)
    # initialize mean/covariance of current sampler
    mu_ais = np.zeros(d)
    mu_ais[0] = 1/np.sqrt(d)
    mu_ais[1] = -1/np.sqrt(d)
    mu_num_ais = np.zeros(d)
    mu_denom_ais = 0
    Cov_ais = (5/d) * ((df-2)/df) * np.eye(d)
    # initialize (w)-AIS estimates
    num_ais_total = 0
    denom_ais_total = 0
    num_wais_total = 0
    denom_wais_total = 0
    # initialize list of results
    I_ais_list = []
    I_wais_list = []
    I_aiscv_list = []
    full_H = np.zeros((0,m))
    full_phi = np.zeros((0,d))
    full_weights = np.zeros(0)

    # main loop
    temp=0
    while temp<T:
        # sample from current sampler q
        𝜃 = multivariate_student_generate(m=mu_ais,S=Cov_ais,df=df,n=n_t)
        grads_log_target = grad_log_f(𝜃=𝜃)
        if not np.isnan(grads_log_target).any():
            # evaluations of integrand g, sampler q, and target f
            eval_g = np.array([func(val) for val in 𝜃])
            eval_q = multivariate_student_pdf(x=𝜃,mean=mu_ais,shape=Cov_ais,df=df)
            eval_f = target(𝜃=𝜃)
            # compute normalizing weights w=f/q
            weights = eval_f/eval_q
            # compute num/denom of AIS estimate
            num_ais_curr = weights@eval_g
            denom = weights.sum()
            # Update num/denom of AIS estimate
            num_ais_total += num_ais_curr
            denom_ais_total += denom
            # compute num/denom of wAIS estimate
            mat_var = 1/np.mean((weights-1)**2)
            num_stab = mat_var * num_ais_curr
            denom_stab = mat_var * denom
            # Update num/denom of wAIS estimate
            num_wais_total += num_stab
            denom_wais_total += denom_stab
            # compute num/denom of new sampler q
            A = np.multiply(𝜃,weights.reshape(n_t,1))
            num = A.sum(axis=0)            
            # update sampler mean
            mu_num_ais += num
            mu_denom_ais += denom
            mu_ais = mu_num_ais/mu_denom_ais
            ## AISCV part
            H = np.zeros((n_t,m))
            for j in range(m):
                alpha = mask[j]

                prods = np.prod(np.power(𝜃,alpha),axis=1)
                diags = alpha/𝜃
                grads = np.array([np.dot(np.diag(diags[k]),prods[k]*np.ones(len(alpha))) for k in range(n_t)])
                diags2 = alpha*(alpha-1)/(𝜃**2)
                laps = np.array([np.dot(np.diag(diags2[k]),prods[k]*np.ones(len(alpha))) for k in range(n_t)]).sum(axis=1)
                H[:,j] = laps + np.multiply(grads,grads_log_target).sum(axis=1)
            # update variables for OLS problem
            full_phi = np.concatenate((full_phi, eval_g))
            full_weights = np.concatenate((full_weights,weights))
            full_H = np.concatenate((full_H,H))
            # compute and save estimates
            I_ais = num_ais_total/denom_ais_total
            I_wais = num_wais_total/denom_wais_total
            temp +=1
        else:
            print("Nan in grad_log_f")
    # Solve OLS problem
    ols = LinearRegression(fit_intercept=True, normalize=True,n_jobs=-1)
    ols.fit(X=full_H,y=full_phi,sample_weight = full_weights)
    I_aiscv = (ols.intercept_)
    return I_ais,I_wais,I_aiscv

def run_multi(func,N_exp,d,Q,T,n_t,df):
    """ Run many experiments of AIS and AISCV on isotropic gaussian mixtures
    Params:
    @func (function): integrand g of the problem
    @N_exp     (int): number of replications
    @d         (int): dimension of the problem
    @Q         (int): total bounded degree for control variates
    @T         (int): number of stages
    @n_t       (int): number of samples at each stage
    @df        (int): degrees of freedom for multivariate student law
    Returns:
    res_ais,res_wais,res_aiscv
    """
    res_ais = np.zeros((N_exp,d))
    res_wais = np.zeros((N_exp,d))
    res_aiscv = np.zeros((N_exp,d))
    for k in tqdm(range(N_exp)):
        I_ais, I_wais, I_aiscv = run(func=func,seed=k,d=d,Q=Q,T=T,n_t=n_t,df=df)
        res_ais[k] = I_ais
        res_wais[k] = I_wais
        res_aiscv[k] = I_aiscv
    return res_ais,res_wais,res_aiscv

## Parameters 
N_exp = 100
n_t = 1000
df = 10

## Run experiments
for Q in [2,3]:
    for d in [4,8]:
        I_ais = np.zeros((5,N_exp,d))
        I_wais = np.zeros((5,N_exp,d))
        I_aiscv = np.zeros((5,N_exp,d))
        for i,T in enumerate([5,10,20,30,50]):
            # instance of MC estimate
            res_ais, res_wais, res_aiscv = run_multi(func=g,N_exp=N_exp,d=d,Q=Q,T=T,n_t=n_t,df=df)
            I_ais[i] =  res_ais
            I_wais[i] =  res_wais
            I_aiscv[i] = res_aiscv
        np.save('res_ais_d{}_Nexp{}_'.format(str(d),str(N_exp))+'isotropic'+'.npy',I_ais)
        np.save('res_wais_d{}_Nexp{}_'.format(str(d),str(N_exp))+'isotropic'+'.npy',I_wais)
        np.save('res_aiscv_d{}_Nexp{}_Q{}_'.format(str(d),str(N_exp),str(Q))+'isotropic'+'.npy',I_aiscv)
