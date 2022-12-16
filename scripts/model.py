#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Author: Rémi LELUC, François PORTIER, Johan SEGERS, Aigerim ZHUMAN
This file contains the class AISCV which implements the control variates
Monte Carlo estimator with Adaptive Importance Sampling
'''
# Import libraries
import numpy as np
from sklearn.linear_model import LinearRegression
from controlvariates import get_H_nkd,mask_pairs
from itertools import product
from tqdm import tqdm_notebook as tqdm
from utils import multivariate_student_generate, multivariate_student_pdf
from utils import multivariate_uniform_pdf

class AISCV():
    ''' Adaptive Importance Sampling Control Variates Monte Carlo estimator  '''
    def __init__(self,func):
        self.func = func   # function to integrate
    
    def run(self,seed,d,T,n_t,sigma_0,df,basis,mode,k):
        ''' Perform one simulation with AIS and AISCV
        Params:
        @seed      (int): random seed for reproducibility
        @d         (int): dimension of the problem
        @T         (int): number of stages
        @n_t       (int): allocation policy (number samples drawn at each stages)
        @sigma_0 (float): variance of covariance for multivariate student distribution
        @df        (int): degrees of freedom of multivariate student distribution
        @basis     (str): basis of polynomial family for control variates
        @mode      (str): 'all' or 'pairs' when building tensor products of control variates
        @k         (int): number of control variates in each direction
        '''
        # compute mask for tensor product of control variates
        if mode=='all':
            m = (k+1)**(d) - 1
            mask = np.delete(arr=sorted(list(product(range(k + 1), repeat=d)), key=sum),
                             obj=0, axis=0)
        elif mode=='pairs':
            m = int(k*d + k*k*d*(d-1)/2)
            mask = mask_pairs(k=k,d=d)

        # set random seed for the run
        np.random.seed(seed)
        # initialize mean/covariance of current sampler
        self.mu_ais = 0.5*np.ones(d)
        self.mu_num_ais = np.zeros(d)
        self.mu_denom_ais = 0
        self.Cov_ais = sigma_0*((df-2)/df)*np.eye(d)
        # initialize (w)-AIS estimates
        self.num_ais_total = 0
        self.denom_ais_total = 0
        self.num_wais_total = 0
        self.denom_wais_total = 0
        # initialize list of results
        add = np.ones((n_t, 1, d))
        full_H = np.zeros((0,m))
        full_phi = np.zeros((0))
        full_weights = np.zeros(0)
        # Loop over number of stages
        for t in range(T):
            # sample from current sampler q
            x = multivariate_student_generate(m=self.mu_ais,S=self.Cov_ais,df=df,n=n_t)
            # evaluations of integrand phi
            eval_phi = np.array([self.func(val) for val in x])
            # evaluations of current sampler q and target distribution f
            eval_q = multivariate_student_pdf(x=x,mean=self.mu_ais,shape=self.Cov_ais,df=df)
            eval_f = multivariate_uniform_pdf(x)
            # compute normalizing weights w=f/q
            weights = eval_f/eval_q
            ## (w)AIS part
            # compute num/denom of AIS estimate
            num_ais_curr = (eval_phi*weights).sum()
            denom = weights.sum()
            # Update num/denom of AIS estimate
            self.num_ais_total += num_ais_curr
            self.denom_ais_total += denom
            # compute num/denom of wAIS estimate
            mat_var = 1/np.mean((weights-1)**2)
            num_stab = mat_var * num_ais_curr
            denom_stab = mat_var * denom
            # Update num/denom of wAIS estimate
            self.num_wais_total += num_stab
            self.denom_wais_total += denom_stab
            # compute num/denom of new sampler q
            X = np.multiply(x,weights.reshape(n_t,1))
            num = X.sum(axis=0)            
            # update sampler mean
            self.mu_num_ais += num
            self.mu_denom_ais += denom
            self.mu_ais = self.mu_num_ais/self.mu_denom_ais
            ## AISCV part
            # Compute matrix H of size n x k x d in given basis
            H_temp = get_H_nkd(X=x, k=k, basis=basis)
            # Add [1,...,1] to represent control h_0 = 1
            H_temp = np.concatenate((add, H_temp), axis=1)
            # Build matrix H of size n_t x m
            H_curr = np.empty((n_t, m))
            for j in range(n_t):  # Fill i-th row
                # compute product of separable variables
                H_curr[j] = np.prod(a=np.choose(mask,H_temp[j]), axis=1)
            # update variables for OLS problem
            full_phi = np.concatenate((full_phi,eval_phi))
            full_weights = np.concatenate((full_weights,weights))
            full_H = np.concatenate((full_H,H_curr))         
            # compute and save estimates
            self.I_ais = self.num_ais_total/self.denom_ais_total
            self.I_wais = self.num_wais_total/self.denom_wais_total
        # Solve OLS at the end
        ols = LinearRegression(fit_intercept=True, normalize=True,n_jobs=-1)
        ols.fit(X=full_H,y=full_phi,sample_weight = full_weights)
        self.I_aiscv = (ols.intercept_)
        
    def run_multi(self,N_exp,d,T,n_t,sigma_0,df,basis,mode,k):
        # display information
        print('d=',d,'; T=',T,'; n_t=',n_t)
        # initialize final results
        self.res_ais = np.zeros(N_exp)
        self.res_wais = np.zeros(N_exp)
        self.res_aiscv = np.zeros(N_exp)
        # loop over number of replications
        for i in tqdm(range(N_exp)):
            self.run(seed=i,d=d,T=T,n_t=n_t,
                     sigma_0=sigma_0,df=df,basis=basis,
                     mode=mode,k=k)
            self.res_ais[i] = self.I_ais
            self.res_wais[i] = self.I_wais
            self.res_aiscv[i] = self.I_aiscv
       
    def get_I_ais_list(self):
        return self.I_ais_list
    
    def get_I_wais_list(self):
        return self.I_wais_list
    
    def get_I_aiscv_list(self):
        return self.I_aiscv_list
    
    
