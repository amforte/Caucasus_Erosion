#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Written by Adam M. Forte for 
Low runoff variability driven by a dominance of snowmelt inhibits clear coupling of climate, tectonics, and topography in the Greater Caucasus Mountains

If you use this code or derivatives, please cite the original paper.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import stochastic_threshold as stim


def rmse(obs,pred):
    return np.sqrt(np.sum((obs-pred)**2)/len(obs))
def wrmse(obs,pred,weights):
    return np.sqrt(np.sum((1/weights**2)*((obs-pred)**2))) 
def powlaw(B,x):
    return B[0]*(x**B[1])


######################
#### Read In Data ####
######################
df=pd.read_csv('data_tables/gc_ero_master_table.csv')
# Extract main variables of interest
mR=df['mean_runoff'].to_numpy()
mP=df['corrected_mean_trmm'].to_numpy()
ksn=df['mean_ksn'].to_numpy()
ksn_u=df['se_ksn'].to_numpy()
e=df['St_E_rate_m_Myr'].to_numpy()
e_u=df['St_Ext_Unc'].to_numpy()
# Choice of k estimation technique
k_var='k_z_est'
k_var1='k_SSN_est'

k_values=df[[k_var,k_var1]]
k=k_values.mean(1).to_numpy()
k_std=k_values.std(1)

# Load in k_e optimization results
dfke=pd.read_csv('result_tables/k_e_optim.csv')
k_e=dfke['k_e_boostrap'].to_numpy()
est_err=dfke['rmse_bootstrap'].to_numpy()
e_predicted=dfke['E_pred'].to_numpy() 

# Load in SPIM fit values
dfSPIM=pd.read_csv('result_tables/w_hi__bootstrap.csv')
K=dfSPIM['K'].to_numpy()[0]
n=dfSPIM['n'].to_numpy()[0]

## Calculate RMSE for median STIM on Erosion Rate  
# Prediction for means of runoff, k, and k_e
k_e_med=np.median(k_e)
k_med=np.median(k)
R_med=np.median(mR)
cL=stim.set_constants(R_med,k_e_med)

E_STIMmed_pred=np.zeros(len(ksn))
for i in range(len(ksn)):
    [E_STIMmed_pred[i],E_err,Q_starc]=stim.stim_one(ksn[i],k_med,cL)

medianSTIM_E_rmse=rmse(e,E_STIMmed_pred)
medianSTIM_E_rmse_w=wrmse(e,E_STIMmed_pred,e_u)

## Estimate RMSE for median STIM on ksn
[Ks,E,E_err,Q_starc]=stim.stim_range(k_med,cL,max_ksn=600,num_points=600)
Ksn_STIMmed_pred=np.zeros(len(ksn))
for i in range(len(ksn)):
    ix=np.argmin(np.abs(E-e[i]))
    Ksn_STIMmed_pred[i]=Ks[ix]
    
medianSTIM_ksn_rmse=rmse(ksn,Ksn_STIMmed_pred)
medianSTIM_ksn_rmse_w=wrmse(ksn,Ksn_STIMmed_pred,ksn_u)

## Calculate RMSE for SPIM on Erosion Rate
E_SPIM_pred=powlaw([K,n],ksn)
C=K**(-1/n)
phi=(1/n)
Ksn_SPIM_pred=powlaw([C,phi],e)

SPIM_E_rmse=rmse(e,E_SPIM_pred)
SPIM_E_rmse_w=wrmse(e,E_SPIM_pred,e_u)
SPIM_ksn_rmse=rmse(ksn,Ksn_SPIM_pred)
SPIM_ksn_rmse_w=wrmse(ksn,Ksn_SPIM_pred,ksn_u)

## Calculate RMSE for fit STIM on Erosion Rate 
# Load in STIM fit
dfSTIM=pd.read_csv('result_tables/w_hi__stim_odr.csv')
stim_R=dfSTIM['R'].to_numpy()[0]
stim_k_e=dfSTIM['k_e'].to_numpy()[0]
stim_k=dfSTIM['k'].to_numpy()[0]
stim_cL=stim.set_constants(stim_R,stim_k_e)

E_STIMfit_pred=np.zeros(len(ksn))
for i in range(len(ksn)):
    [E_STIMfit_pred[i],E_err,Q_starc]=stim.stim_one(ksn[i],stim_k,stim_cL)

fitSTIM_E_rmse=rmse(e,E_STIMfit_pred)
fitSTIM_E_rmse_w=wrmse(e,E_STIMfit_pred,e_u)

## Estimate RMSE for median STIM on ksn
[KsF,EF,E_errF,Q_starcF]=stim.stim_range(stim_k,stim_cL,max_ksn=600,num_points=600)
Ksn_STIMfit_pred=np.zeros(len(ksn))
for i in range(len(ksn)):
    ix=np.argmin(np.abs(EF-e[i]))
    Ksn_STIMfit_pred[i]=KsF[ix]
    
fitSTIM_ksn_rmse=rmse(ksn,Ksn_STIMfit_pred)
fitSTIM_ksn_rmse_w=wrmse(ksn,Ksn_STIMfit_pred,ksn_u)

## Make Figure
fig1=plt.figure(num=1,figsize=(20,15))
ax1=plt.subplot(2,3,1)
ax1.stem(e,ksn-Ksn_SPIM_pred,label='RMSE = {:2.2f}, WRMSE = {:2.2f}'.format(SPIM_ksn_rmse,SPIM_ksn_rmse_w))
ax1.set_ylabel('Residual on $k_{sn}$ [m]')
ax1.set_xlabel('Erosion Rate [m/Myr]')
ax1.set_title('SPIM')
ax1.legend(loc='best')
ax1.set_xscale('log')
ax1.set_ylim((-275,210))

ax3=plt.subplot(2,3,2)
ax3.stem(e,ksn-Ksn_STIMmed_pred,label='RMSE = {:2.2f}, WRMSE = {:2.2f}'.format(medianSTIM_ksn_rmse,medianSTIM_ksn_rmse_w))
ax3.set_ylabel('Residual on $k_{sn}$ [m]')
ax3.set_xlabel('Erosion Rate [m/Myr]')
ax3.set_title('STIM - Median')
ax3.legend(loc='best')
ax3.set_xscale('log')
ax3.set_ylim((-275,210))

ax5=plt.subplot(2,3,3)
ax5.stem(e,ksn-Ksn_STIMfit_pred,label='RMSE = {:2.2f}, WRMSE = {:2.2f}'.format(fitSTIM_ksn_rmse,fitSTIM_ksn_rmse_w))
ax5.set_ylabel('Residual on $k_{sn}$ [m]')
ax5.set_xlabel('Erosion Rate [m/Myr]')
ax5.set_title('STIM - Fit')
ax5.legend(loc='best')
ax5.set_xscale('log')
ax5.set_ylim((-275,210))

ax2=plt.subplot(2,3,4)
ax2.stem(ksn,e-E_SPIM_pred,label='RMSE = {:2.2f}, WRMSE = {:2.2f}'.format(SPIM_E_rmse,SPIM_E_rmse_w))
ax2.set_ylabel('Residual on E [m/Myr]')
ax2.set_xlabel('$k_{sn}$ [m]')
ax2.set_title('SPIM')
ax2.legend(loc='best')
ax2.set_ylim((-3500,5300))

ax4=plt.subplot(2,3,5)
ax4.stem(ksn,e-E_STIMmed_pred,label='RMSE = {:2.2f}, WRMSE = {:2.2f}'.format(medianSTIM_E_rmse,medianSTIM_E_rmse_w))
ax4.set_ylabel('Residual on E [m/Myr]')
ax4.set_xlabel('$k_{sn}$ [m]')
ax4.set_title('STIM - Median')
ax4.legend(loc='best')
ax4.set_ylim((-3500,5300))

ax6=plt.subplot(2,3,6)
ax6.stem(ksn,e-E_STIMfit_pred,label='RMSE = {:2.2f}, WRMSE = {:2.2f}'.format(fitSTIM_E_rmse,fitSTIM_E_rmse_w))
ax6.set_ylabel('Residual on E [m/Myr]')
ax6.set_xlabel('$k_{sn}$ [m]')
ax6.set_title('STIM - Fit')
ax6.legend(loc='best')
ax6.set_ylim((-3500,5300))

