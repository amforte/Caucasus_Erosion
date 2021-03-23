#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Written by Adam M. Forte for 
Low runoff variability driven by a dominance of snowmelt inhibits clear coupling of climate, tectonics, and topography in the Greater Caucasus Mountains

If you use this code or derivatives, please cite the original paper.
"""
import pandas as pd
import numpy as np
from scipy import odr
import matplotlib.pyplot as plt

import stochastic_threshold as stim

######################
#### Read In Data ####
######################

df=pd.read_csv('data_tables/gc_ero_master_table.csv')
# Extract main variables of interest
mR=df['mean_runoff'].to_numpy()
ksn=df['mean_ksn'].to_numpy()
ksn_u=df['se_ksn'].to_numpy()
e=df['St_E_rate_m_Myr'].to_numpy()
e_u=df['St_Ext_Unc'].to_numpy()
chi_r2=df['chi_R_squared'].to_numpy()
# Choice of k estimation technique
k_var='k_z_est'
k_var1='k_SSN_est'

# Calculated mean k values
k_values=df[[k_var,k_var1]]
k=k_values.mean(1).to_numpy()

# Load in k_e optimization results
dfke=pd.read_csv('result_tables/k_e_optim.csv')
k_e=dfke['k_e_boostrap'].to_numpy()

# Use Medians as starting values
k_e_med=np.median(k_e)
k_med=np.median(k)
R_med=np.median(mR)

# Flag for data exclusion rule
exclude_high=True

# Set strings
if exclude_high:
    hi_dat='no_hi'
else:
    hi_dat='w_hi'

# Build prefix
f_pre=hi_dat+'__'

# Output Location
out_dir='result_tables/'

########################
#### Data Exclusion ####
########################
if exclude_high:
    idx=np.logical_and(e<2500,chi_r2>0.90) 
    ksn_ex=ksn[np.invert(idx)]
    ksn_u_ex=ksn_u[np.invert(idx)]
    e_ex=e[np.invert(idx)]
    e_u_ex=e_u[np.invert(idx)]
    ksn=ksn[idx]
    e=e[idx]
    ksn_u=ksn_u[idx]
    e_u=e_u[idx]
else:
    idx=e_u<e
    ksn_ex=ksn[np.invert(idx)]
    ksn_u_ex=ksn_u[np.invert(idx)]
    e_ex=e[np.invert(idx)]
    e_u_ex=e_u[np.invert(idx)]
    ksn=ksn[idx]
    e=e[idx]
    ksn_u=ksn_u[idx]
    e_u=e_u[idx]  

# Generate real data input without uncertainty
Rdata=odr.RealData(ksn,e)

# Generate fit object
def stim_odr(B,x):
    cLi=stim.set_constants(B[0],B[1],omega_s=0.25)
    E=np.zeros(x.shape)
    for i in range(len(x)):
        [E[i],E_err,Q_starc]=stim.stim_one(x[i],B[2],cLi)
    return E

# Build and run ODR fit
stim_model=odr.Model(stim_odr)
stim_odr=odr.ODR(Rdata,stim_model,beta0=[R_med,k_e_med,k_med])
stim_res=stim_odr.run()

# Extract fit parameters
stim_R=stim_res.beta[0]
stim_k_e=stim_res.beta[1]
stim_k=stim_res.beta[2]

# Package
out_df=pd.DataFrame({'R':[stim_R],'k_e':[stim_k_e],'k':[stim_k],})
out_df.to_csv(out_dir+f_pre+'stim_odr.csv',index=False)

# Generate STIM relationships for median and fit
cL=stim.set_constants(R_med,k_e_med)
[Ks,E,E_err,Q_starc]=stim.stim_range(k_med,cL,max_ksn=600,num_points=500)
stim_cL=stim.set_constants(stim_R,stim_k_e)
[stim_Ks,stim_E,E_err,Q_starc]=stim.stim_range(stim_k,stim_cL,max_ksn=600,num_points=500)

# PLot Outputs
fig1=plt.figure(num=1,figsize=(8,15))
ax1=plt.subplot(2,1,1)
# Plot data used in fit
ax1.scatter(e,ksn,s=40,c='black',zorder=3)
ax1.errorbar(e,ksn,yerr=ksn_u,xerr=e_u,
              linestyle='none',c='black',zorder=2,elinewidth=0.5)
# Data excluded from fit
ax1.scatter(e_ex,ksn_ex,s=40,c='grey',zorder=3)
ax1.errorbar(e_ex,ksn_ex,yerr=ksn_u_ex,xerr=e_u_ex,
              linestyle='none',c='grey',zorder=2,elinewidth=0.5)  
# Plot lines
ax1.plot(E,Ks,c='k',linestyle='-',linewidth=2,label='STIM: Median Values')
ax1.plot(stim_E,stim_Ks,c='k',linestyle=':',linewidth=2,label='STIM: ODR Fit')
plt.xlim((0,8000))
plt.ylim((0,600))
plt.xlabel('Erosion Rate [m/Myr]')
plt.ylabel('$k_{sn}$ [m]')
plt.legend(loc='best')

ax2=plt.subplot(2,1,2)
# Plot data used in fit
ax2.scatter(e,ksn,s=40,c='black',zorder=3)
ax2.errorbar(e,ksn,yerr=ksn_u,xerr=e_u,
              linestyle='none',c='black',zorder=2,elinewidth=0.5)
# Data excluded from fit
ax2.scatter(e_ex,ksn_ex,s=40,c='grey',zorder=3)
ax2.errorbar(e_ex,ksn_ex,yerr=ksn_u_ex,xerr=e_u_ex,
              linestyle='none',c='grey',zorder=2,elinewidth=0.5)  
# Plot lines
ax2.plot(E,Ks,c='k',linestyle='-',linewidth=2,label='STIM: Median Values')
ax2.plot(stim_E,stim_Ks,c='k',linestyle=':',linewidth=2,label='STIM: ODR Fit')
plt.xlim((10,10**4))
plt.ylim((0,600))
plt.xlabel('Erosion Rate [m/Myr]')
plt.ylabel('$k_{sn}$ [m]')
plt.legend(loc='best')
plt.xscale('log')