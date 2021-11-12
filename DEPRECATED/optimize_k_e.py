#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Written by Adam M. Forte for 
Low runoff variability driven by a dominance of snowmelt inhibits clear coupling of climate, tectonics, and topography in the Greater Caucasus Mountains

If you use this code or derivatives, please cite the original paper.
"""

import pandas as pd
import numpy as np
from scipy.optimize import minimize_scalar
from astropy.utils import NumpyRNGContext

import stochastic_threshold as stim

# Read in data
df=pd.read_csv('data_tables/gc_ero_master_table.csv')

mR=df['mean_runoff'].to_numpy()
ksn=df['mean_ksn'].to_numpy()
ksn_u=df['se_ksn'].to_numpy()
e=df['St_E_rate_m_Myr'].to_numpy()
e_u=df['St_Ext_Unc'].to_numpy()
# Choice of k estimation technique
k_var='k_z_est'
k_var1='k_SSN_est'

# Assigned k values
k_values=df[[k_var,k_var1]]
k=k_values.mean(1).to_numpy()
k_std=k_values.std(1)

# For reproducibility
seed=5

## OPTIMIZATION SCHEME FOR k_e
k_e=np.zeros(mR.shape)
k_e_simple=np.zeros(mR.shape)
est_err=np.zeros(mR.shape)
e_predicted=np.zeros(mR.shape)
num_synth=500
with NumpyRNGContext(seed):
    for i in range(len(mR)):
        mR_val=mR[i]
        k_val=k[i]
        ksn_val=ksn[i]
        e_val=e[i]
        # Generate distributions
        ksn_dist=np.random.normal(ksn[i],ksn_u[i],num_synth)
        e_dist=np.random.normal(e[i],e_u[i],num_synth)
        
        
        def optim_k_e(x):
            con_list=stim.set_constants(mR_val,x)
            [E_pred,E_err,Q_starc]=stim.stim_one(ksn_val,k_val,con_list)
            err=np.abs(E_pred-e_val)
            return err
        
        def optim_k_e_dist(x):
            con_list=stim.set_constants(mR_val,x)
            E_pred=np.zeros((num_synth))
            for k in range(num_synth):
                [E_pred[k],E_err,Q_starc]=stim.stim_one(ksn_dist[i],k_val,con_list)
            rmse=np.sqrt(np.sum((e_dist-E_pred)**2)/num_synth)
            return rmse
          
        res=minimize_scalar(optim_k_e,bounds=[1e-20,1e-6],method='bounded',
                            options={'maxiter':500000,'xatol':1e-20})
        res_alt=minimize_scalar(optim_k_e_dist,bounds=[1e-20,1e-6],method='bounded',
                                options={'maxiter':500000,'xatol':1e-15})
        k_e_simple[i]=res.x
        k_e[i]=res_alt.x
        est_err[i]=res_alt.fun
        cL=stim.set_constants(mR_val,k_e[i])
        [e_predicted[i],E_err,Q_starc]=stim.stim_one(ksn_val,k_val,cL)
        
        prog=np.round(i/len(mR)*100,2)
        print('Current progress: '+str(prog)+'%')
    
k_e_simple=k_e_simple.reshape((len(mR),1))
k_e=k_e.reshape((len(mR),1))
est_err=est_err.reshape((len(mR),1))
e_predicted=e_predicted.reshape((len(mR),1))

data=np.concatenate((k_e_simple,k_e,est_err,e_predicted),axis=1)
dfout=pd.DataFrame(data,columns=['k_e_simple','k_e_boostrap','rmse_bootstrap','E_pred'])
dfout.to_csv('result_tables/k_e_optim.csv',index=False)
