#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author: Rémi LELUC, François PORTIER, Johan SEGERS, Aigerim ZHUMAN
This file contains the functions to perform Bayesian Linear Regression.
'''

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_svmlight_file
from tqdm import tqdm
import torch
from torch import tensor, eye, sqrt, zeros, log, exp, ones
from torch.distributions import MultivariateNormal as MVN

from utils import multivariate_student_generate, multivariate_student_pdf
from utils import get_comb_lower
from scipy.sparse import csr_matrix

def load_data(filename):
    S,y = load_svmlight_file(f=filename)
    X = csr_matrix.toarray(S)
    return X,y

import matplotlib.pyplot as plt


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

# Data Gram matrix
G = tensor((X.T).dot(X))
print("G.shape: ",G.shape)

torch.random.manual_seed(42)
mu_star = torch.zeros(n_features).reshape(-1,1)
sigma_y = 50
cov_star = (sigma_y**2)*eye(n_features)

# prior gaussian
MVN_un = MVN(mu_star.ravel(),cov_star)

res = torch.linalg.solve(cov_star,mu_star)
Σs = torch.inverse((sigma_y**(-2)) * G + torch.inverse(cov_star))
μs = torch.matmul(Σs,(sigma_y**(-2)) * tensor((X.T).dot(y)) + res.ravel() )

# posterior gaussian
MVN_U = MVN(μs,Σs)

# integrand
def g(𝜃):
    return (𝜃.T).dot(𝜃)

#unnormalized target distribution
def target_u(𝜃):
    p = len(𝜃)
    #print(p)\n",
    𝜃 = tensor(𝜃,requires_grad=False).double()
    prior = exp(MVN_un.log_prob(𝜃).double())
    data_term = - np.sum((y-np.dot(X,𝜃))**2)
    LL = ((sigma_y**(-2))/2) * data_term - (p/2)*np.log(2*np.pi)
    return prior.detach().numpy() * np.exp(LL) 

def grad_log_prior(𝜃):
    p = tensor(𝜃.shape[1])
    𝜃 = tensor(𝜃,requires_grad=True).double()
    log_prior = log(exp(MVN_un.log_prob(𝜃).double()))
    # compute gradient
    log_prior.backward(torch.ones(𝜃.shape[0]))
    return (𝜃.grad).detach().numpy()

def grad_log_likelihood(𝜃):
    err = y-np.dot(X,𝜃)
    return ((sigma_y**(-2))/2)*np.dot(X.T,err)/n_samples

def target(𝜃):
    p = tensor(𝜃.shape[1])
    𝜃 = tensor(𝜃,requires_grad=False).double()
    eval_f = exp(MVN_U.log_prob(𝜃).double())
    return eval_f.detach().numpy()
    
def grad_log_f(𝜃):
    p = tensor(𝜃.shape[1])
    𝜃 = tensor(𝜃,requires_grad=True).double()
    log_f = log(exp(MVN_U.log_prob(𝜃).double()))
    # compute gradient
    log_f.backward(torch.ones(𝜃.shape[0]))
    return (𝜃.grad).detach().numpy()

def run_stein_BLR(func,seed,d,Q,T,n_t,df):
    """ Run AIS and AISCV for Bayesian Linear Regression problem
    with Stein Control variates with total degree bounded by Q
    @func (function): integrand g of the problem
    @seed      (int): random seed for reproducibility
    @d         (int): dimension of the problem
    @Q         (int): total bounded degree for control variates
    @T         (int): number of stages
    @n_t       (int): number of samples at each stage (allocation policy)
    @df        (int): degrees of freedom for multivariate student law
    """
    # Create all permutations of powers alphas\n",
    mask = get_comb_lower(Q=Q,size=d)
    m = len(mask)
    # set random seed for the run
    np.random.seed(seed)
    # initialize mean/covariance of current sampler
    mu_ais = np.zeros(d)
    mu_num_ais = np.zeros(d)
    mu_denom_ais = 0
    Cov_ais = np.array(Σs) 
    # initialize (w)-AIS estimates
    num_ais_total = 0
    denom_ais_total = 0
    num_wais_total = 0
    denom_wais_total = 0
    num_cov_ais = 0
    # initialize list of results
    I_ais_list = []
    I_aiscv_list = []
        
    # multi-dimensional integrand g
    #full_H = np.zeros((0,m))
    #full_phi = np.zeros((0,d))
    #full_weights = np.zeros(0)

    # univariate integrand g
    full_H = np.zeros((0,m))
    full_phi = np.zeros(0)
    full_weights = np.zeros(0)
    # main loop\n",
    temp=0
    while temp<T:
        # sample from current sampler q
        𝜃 = multivariate_student_generate(m=mu_ais,S=Cov_ais,df=df,n=n_t)
        grads_log_target = grad_log_f(𝜃=𝜃)
        if not np.isnan(grads_log_target).any():
            # evaluations of integrand g, sampler q, and target f
            eval_g = np.array([func(val) for val in 𝜃])
            #print(\"eval_g = \",eval_g)
            eval_q = multivariate_student_pdf(x=𝜃,mean=mu_ais,shape=Cov_ais,df=df)
            eval_f = np.array([target_u(𝜃=𝜃_c) for 𝜃_c in 𝜃])
            #eval_f = target(𝜃=𝜃)
            #print('eval_f =', eval_f)
            # compute normalizing weights w=f/q
            weights = eval_f/eval_q
            # compute num/denom of AIS estimate
            num_ais_curr = weights@eval_g
            denom = weights.sum()
            # Update num/denom of AIS estimate
            num_ais_total += num_ais_curr
            denom_ais_total += denom
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
            temp +=1
        else:
            print("Nan in grad_log_f")
    # Solve OLS problem
    ols = LinearRegression(fit_intercept=True, normalize=True,n_jobs=-1)
    ols.fit(X=full_H,y=full_phi,sample_weight = full_weights)
    I_aiscv = (ols.intercept_)
    return I_ais,I_aiscv


def run_multi(func,N_exp,d,Q,T,n_t,df):
    """ Run multiple AIS and AISCV for Bayesian Linear Regression problem
    with Stein Control variates with total degree bounded by Q
    @func (function): integrand g of the problem
    @N_exp     (int): number of replications
    @d         (int): dimension of the problem
    @Q         (int): total bounded degree for control variates
    @T         (int): number of stages
    @n_t       (int): number of samples at each stage (allocation policy)
    @df        (int): degrees of freedom for multivariate student law
    """
    res_ais = np.zeros(N_exp)
    res_aiscv = np.zeros(N_exp)
    for k in tqdm(range(N_exp)):
        I_ais, I_aiscv = run_stein_BLR(func=func,seed=k,d=d,Q=Q,T=T,n_t=n_t,df=df)
        res_ais[k] = I_ais
        res_aiscv[k] = I_aiscv
    return res_ais, res_aiscv

## Parameters
N_exp = 100
n_t = 1000
df = 10

## Run experiments
for Q in [1,2]:
    I_ais = np.zeros((5,N_exp))
    I_aiscv = np.zeros((5,N_exp))
    for i,T in enumerate([5,10,20,30,50]):
        # instance of MC estimate
        res_ais, res_aiscv = run_multi(func=g,N_exp=N_exp,d=d,Q=Q,T=T,n_t=n_t,df=df)
        I_ais[i] =  res_ais
        I_aiscv[i] = res_aiscv
    np.save('res_ais_housing_Nexp{}_Q{}'.format(str(N_exp),str(Q))+'.npy',I_ais)
    np.save('res_aiscv_housing_Nexp{}_Q{}'.format(str(N_exp),str(Q))+'.npy',I_aiscv)
 
