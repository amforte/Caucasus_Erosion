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
from sklearn.neighbors import KernelDensity
from astropy.utils import NumpyRNGContext
from astropy.stats import bootstrap

import stochastic_threshold as stim

########################
#### Set Parameters ####
########################

# For reproducibility
seed=5

# Switch to exclude ero rates above 2500 m/Myr from fits
exclude_high=True

#### Main switch to run specific algorithms ####
# Monte-carlo based sampling - for each replicate, will produce a synthetic
# dataset with the same number of samples as the original dataset, where values
# for each basin are derived from random sampling constrained by the mean and
# std of the erosion rate and ksn for that basin (assuming gaussian)
run_mc=True
# Bootstrap based sampling - for each replicate, will produce a synthetic 
# dataset with the same number of samples as the original dataset created by
# bootstrap sampling (i.e. sampling with replacement) and use the observed
# uncertainties for each basin
run_bs=True


# For All Routines
num_replicates=1e6
num_replicates=int(num_replicates)

# Output Location
out_dir='result_tables/'

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

k_values=df[[k_var,k_var1]]
k=k_values.mean(1).to_numpy()
k_std=k_values.std(1)


# Set strings
if exclude_high:
    hi_dat='no_hi'
else:
    hi_dat='w_hi'

# Build prefix
f_pre=hi_dat+'__'
    
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

#####################
#### Simple Fits ####
#####################
def powlaw(B,x):
    return B[0]*(x**B[1])

## Orthogonal distance regression for log-transformed linear
def lin(B,x):
    return B[1]*x+B[0]

ll_model=odr.Model(lin)

def rmse(obs,pred):
    return np.sqrt(np.sum((obs-pred)**2)/len(obs))


# Load in k_e optimization results
dfke=pd.read_csv('result_tables/k_e_optim.csv')
k_e=dfke['k_e_boostrap'].to_numpy()
est_err=dfke['rmse_bootstrap'].to_numpy()
e_predicted=dfke['E_pred'].to_numpy() 
   
# Prediction for means of runoff, k, and k_e
k_e_mean=np.median(k_e)
k_mean=np.median(k)
R_mean=np.median(mR)
cL=stim.set_constants(R_mean,k_e_mean)
[Ks,E,E_err,Q_starc]=stim.stim_range(k_mean,cL,max_ksn=600,num_points=500)

# Generate KSN
ksn_vec=np.arange(0,1000,5)

########################
#### Run Procedures ####
########################
if run_mc:
    ##########################
    #### Monte Carlo Fits ####
    ##########################
    # Initiate arrays
    mcK=np.zeros((num_replicates))
    mcn=np.zeros((num_replicates))
    mcss=np.zeros((num_replicates))
    mcsss=np.zeros((num_replicates))
    
    with NumpyRNGContext(seed):
        for i in range(num_replicates):
            ksn_smpl=np.random.normal(ksn,ksn_u)
            e_smpl=np.random.normal(e,e_u)
            while any(ksn_smpl<=0) or any(e_smpl<=0):
                ksn_smpl=np.random.normal(ksn,ksn_u)
                e_smpl=np.random.normal(e,e_u) 
            ll_data=odr.RealData(np.log10(e_smpl),np.log10(ksn_smpl))
            llodr_obj=odr.ODR(ll_data,ll_model,beta0=[3.1,0.4])
            llres_ins=llodr_obj.run()
            lls_C=10**llres_ins.beta[0]
            lls_phi=llres_ins.beta[1]
            mcK[i]=lls_C**(-1/lls_phi)
            mcn[i]=1/lls_phi
            # Calculate RMSE for fit with original data
            mcss[i]=rmse(ksn,powlaw([lls_C,lls_phi],e))
            # Calculate RMSE for fit with sampled data
            mcsss[i]=rmse(ksn_smpl,powlaw([lls_C,lls_phi],e_smpl)) 
            if (i)%(num_replicates/10)==0:
                print(str(i)+' Monte-Carlo runs completed')            

if run_bs:
    ########################
    #### Bootstrap Fits ####
    ########################
    # Initiate arrays
    bsK=np.zeros((num_replicates))
    bsn=np.zeros((num_replicates))
    bsss=np.zeros((num_replicates))
    bssss=np.zeros((num_replicates))
    
    with NumpyRNGContext(seed):
        # Generate indices to sample
        inds=np.arange(0,len(e),1)
        # Generate bootstrapped indices
        ind_bs=bootstrap(inds,num_replicates)
        ind_bs=ind_bs.astype('int64')
        for i in range(num_replicates):
            ksn_smpl=ksn[ind_bs[i,:]]
            e_smpl=e[ind_bs[i,:]]
            ksn_err=ksn_u[ind_bs[i,:]]
            e_err=e_u[ind_bs[i,:]]
            ll_data=odr.RealData(np.log10(e_smpl),np.log10(ksn_smpl),
                                     np.log10(e_err),np.log10(ksn_err))

            llodr_obj=odr.ODR(ll_data,ll_model,beta0=[3.1,0.4])
            llres_ins=llodr_obj.run()
            lls_C=10**llres_ins.beta[0]
            lls_phi=llres_ins.beta[1]
            bsK[i]=lls_C**(-1/lls_phi)
            bsn[i]=1/lls_phi
            # Calculate RMSE for fit with original data
            bsss[i]=rmse(ksn,powlaw([lls_C,lls_phi],e))
            # Calculate RMSE for fit with sampled data
            bssss[i]=rmse(ksn_smpl,powlaw([lls_C,lls_phi],e_smpl))           

            if (i)%(num_replicates/10)==0:
                print(str(i)+' Bootstrap runs completed')

      

###############
#### Plots ####
###############
mc_col='red'
bs_col='deepskyblue'

#### Histogram and KDE ####
fig2=plt.figure(num=2,figsize=(8,15))

f2ax1=plt.subplot(2,1,1)
# kx_d=np.linspace(-20,-1,1000)
# bins=np.linspace(-20,-1,200)
kx_d=np.linspace(-15,-1,1000)
bins=np.linspace(-15,-1,200)

if run_mc:
    kkde1=KernelDensity(bandwidth=0.25, kernel='gaussian')
    kkde1.fit(np.log10(mcK[:,None]))
    klogprob1=kkde1.score_samples(kx_d[:,None])
    f2ax1.plot(kx_d, np.exp(klogprob1), color=mc_col,linewidth=2)
    f2ax1.hist(np.log10(mcK),alpha=0.5,bins=bins,
             density=True,color=mc_col)
if run_bs:
    kkde1=KernelDensity(bandwidth=0.25, kernel='gaussian')
    kkde1.fit(np.log10(bsK[:,None]))
    klogprob1=kkde1.score_samples(kx_d[:,None])
    f2ax1.plot(kx_d, np.exp(klogprob1), color=bs_col,linewidth=2)
    f2ax1.hist(np.log10(bsK),alpha=0.5,bins=bins,
             density=True,color=bs_col)
    
plt.xlim((np.amin(kx_d),np.amax(kx_d)))
plt.xlabel('Log K Values')
plt.title('Parameter Ranges')

f2ax2=plt.subplot(2,1,2)
# nx_d=np.linspace(1,10,1000)
# bins=np.linspace(1,10,200)
nx_d=np.linspace(1,7,1000)
bins=np.linspace(1,7,200)
if run_mc:
    nkde1=KernelDensity(bandwidth=0.1, kernel='gaussian')
    nkde1.fit(mcn[:,None])
    nlogprob1=nkde1.score_samples(nx_d[:,None])
    f2ax2.plot(nx_d, np.exp(nlogprob1), color=mc_col,linewidth=2)
    f2ax2.hist(mcn,alpha=0.5,label='Monte Carlo Estimation',bins=bins,
             density=True,color=mc_col)
if run_bs:
    nkde1=KernelDensity(bandwidth=0.1, kernel='gaussian')
    nkde1.fit(bsn[:,None])
    nlogprob1=nkde1.score_samples(nx_d[:,None])
    f2ax2.plot(nx_d, np.exp(nlogprob1), color=bs_col,linewidth=2)
    f2ax2.hist(bsn,alpha=0.5,label='Bootstrap Estimation',bins=bins,
             density=True,color=bs_col)    
plt.xlim((np.amin(nx_d),np.amax(nx_d)))    
plt.xlabel('n values')
plt.legend(loc='best')
    
#### Blox Plot ####
fig3=plt.figure(num=3,figsize=(8,4))
f3ax1=plt.subplot(2,1,1)
if run_mc:
    # f3ax1.boxplot(np.log10(mcK),vert=False,positions=[1],
    #             medianprops={'color':mc_col},flierprops={'marker':'.','markersize':0.1})
    f3ax1.boxplot(np.log10(mcK),vert=False,positions=[1],
                medianprops={'color':mc_col},showfliers=False)    
if run_bs:
    # f3ax1.boxplot(np.log10(bsK),vert=False,positions=[2],
    #             medianprops={'color':bs_col},flierprops={'marker':'.','markersize':0.1})
    f3ax1.boxplot(np.log10(bsK),vert=False,positions=[2],
                medianprops={'color':bs_col},showfliers=False)       
plt.xlim((np.amin(kx_d),np.amax(kx_d)))
plt.xlabel('Log K Values')
    
f3ax2=plt.subplot(2,1,2)
if run_mc:
    # plt.boxplot(mcn,vert=False,positions=[1],
    #             medianprops={'color':mc_col},flierprops={'marker':'.','markersize':0.1})
    plt.boxplot(mcn,vert=False,positions=[1],
                medianprops={'color':mc_col},showfliers=False)    
if run_bs:
    # plt.boxplot(bsn,vert=False,positions=[2],
    #             medianprops={'color':bs_col},flierprops={'marker':'.','markersize':0.1}) 
    plt.boxplot(bsn,vert=False,positions=[2],
                medianprops={'color':bs_col},showfliers=False)         
plt.xlim((np.amin(nx_d),np.amax(nx_d))) 
plt.xlabel('n values')
    
#### E-KSN Relationship ####
if run_mc:
    mcKpw_median=np.median(mcK)
    mcnpw_median=np.median(mcn)
    mcKpw_q=np.percentile(mcK,[25,75])
    mcnpw_q=np.percentile(mcn,[25,75])
    pw_mc_ero_vec=powlaw([mcKpw_median,mcnpw_median],ksn_vec)
    pw_mc_ero_vec1=powlaw([mcKpw_q[1],mcnpw_q[0]],ksn_vec)
    pw_mc_ero_vec2=powlaw([mcKpw_q[0],mcnpw_q[1]],ksn_vec)
    # Package
    out_df=pd.DataFrame({'K':[mcKpw_median],'K_q25':[mcKpw_q[0]],'K_q75':[mcKpw_q[1]],
                         'n':[mcnpw_median],'n_q25':[mcnpw_q[0]],'n_q75':[mcnpw_q[1]],
                         'num_reps':[num_replicates]})
    out_df.to_csv(out_dir+f_pre+'monte_carlo.csv',index=False)
if run_bs:
    bsKpw_median=np.median(bsK)
    bsnpw_median=np.median(bsn)
    bsKpw_q=np.percentile(bsK,[25,75])
    bsnpw_q=np.percentile(bsn,[25,75])
    pw_bs_ero_vec=powlaw([bsKpw_median,bsnpw_median],ksn_vec)
    pw_bs_ero_vec1=powlaw([bsKpw_q[1],bsnpw_q[0]],ksn_vec)
    pw_bs_ero_vec2=powlaw([bsKpw_q[0],bsnpw_q[1]],ksn_vec)
    # Package
    out_df=pd.DataFrame({'K':[bsKpw_median],'K_q25':[bsKpw_q[0]],'K_q75':[bsKpw_q[1]],
                         'n':[bsnpw_median],'n_q25':[bsnpw_q[0]],'n_q75':[bsnpw_q[1]],
                         'num_reps':[num_replicates]})
    out_df.to_csv(out_dir+f_pre+'bootstrap.csv',index=False)    
     
# Basic plots of fits
fig4=plt.figure(num=4,figsize=(8,15))
f4ax1=plt.subplot(3,1,1)
f4ax1.plot(E,Ks,c='black',linestyle=':',zorder=1,label='STIM Using Median Values')

   
if run_mc:
    f4ax1.plot(pw_mc_ero_vec,ksn_vec,c=mc_col,zorder=1,label='MC-ODR Fit')
    f4ax1.fill_betweenx(ksn_vec,pw_mc_ero_vec1,pw_mc_ero_vec2,color=mc_col,alpha=0.25)
if run_bs:
    f4ax1.plot(pw_bs_ero_vec,ksn_vec,c=bs_col,zorder=1,label='BS-ODR Fit')
    f4ax1.fill_betweenx(ksn_vec,pw_bs_ero_vec1,pw_bs_ero_vec2,color=bs_col,alpha=0.25)  
# Plot data used in fit
f4ax1.scatter(e,ksn,s=40,c='black',zorder=3)
f4ax1.errorbar(e,ksn,yerr=ksn_u,xerr=e_u,
              linestyle='none',c='black',zorder=2,elinewidth=0.5)
# Data excluded from fit
f4ax1.scatter(e_ex,ksn_ex,s=40,c='grey',zorder=3)
f4ax1.errorbar(e_ex,ksn_ex,yerr=ksn_u_ex,xerr=e_u_ex,
              linestyle='none',c='grey',zorder=2,elinewidth=0.5)    
plt.xlim((0,8000))
plt.ylim((0,600))
plt.xlabel('Erosion Rate [m/Myr]')
plt.ylabel('$k_{sn}$ [m]')
plt.legend(loc='lower right')
plt.title('Number of Replicates : '+str(num_replicates))

f4ax2=plt.subplot(3,1,2)
f4ax2.plot(E,Ks,c='black',linestyle=':',zorder=1,label='STIM Using Median Values')

    
if run_mc:
    f4ax2.plot(pw_mc_ero_vec,ksn_vec,c=mc_col,zorder=1,label='MC-ODR Fit')
    f4ax2.fill_betweenx(ksn_vec,pw_mc_ero_vec1,pw_mc_ero_vec2,color=mc_col,alpha=0.25)
if run_bs:
    f4ax2.plot(pw_bs_ero_vec,ksn_vec,c=bs_col,zorder=1,label='BS-ODR Fit')
    f4ax2.fill_betweenx(ksn_vec,pw_bs_ero_vec1,pw_bs_ero_vec2,color=bs_col,alpha=0.25)   
# Plot data used in fit
f4ax2.scatter(e,ksn,s=40,c='black',zorder=3)
f4ax2.errorbar(e,ksn,yerr=ksn_u,xerr=e_u,
              linestyle='none',c='black',zorder=2,elinewidth=0.5)
# Data excluded from fit
f4ax2.scatter(e_ex,ksn_ex,s=40,c='grey',zorder=3)
f4ax2.errorbar(e_ex,ksn_ex,yerr=ksn_u_ex,xerr=e_u_ex,
              linestyle='none',c='grey',zorder=2,elinewidth=0.5) 
plt.xlim((0,2500))
plt.ylim((0,600))
plt.xlabel('Erosion Rate [m/Myr]')
plt.ylabel('$k_{sn}$ [m]')
plt.legend(loc='lower right')

f4ax3=plt.subplot(3,1,3)
f4ax3.plot(E,Ks,c='black',linestyle=':',zorder=1,label='STIM Using Median Values')


if run_mc:
    f4ax3.plot(pw_mc_ero_vec,ksn_vec,c=mc_col,zorder=1,label='MC-ODR Fit')
    f4ax3.fill_betweenx(ksn_vec,pw_mc_ero_vec1,pw_mc_ero_vec2,color=mc_col,alpha=0.25)
if run_bs:
    f4ax3.plot(pw_bs_ero_vec,ksn_vec,c=bs_col,zorder=1,label='BS-ODR Fit')
    f4ax3.fill_betweenx(ksn_vec,pw_bs_ero_vec1,pw_bs_ero_vec2,color=bs_col,alpha=0.25)
# Plot data used in fit
f4ax3.scatter(e,ksn,s=40,c='black',zorder=3)
f4ax3.errorbar(e,ksn,yerr=ksn_u,xerr=e_u,
              linestyle='none',c='black',zorder=2,elinewidth=0.5)
# Data excluded from fit
f4ax3.scatter(e_ex,ksn_ex,s=40,c='grey',zorder=3)
f4ax3.errorbar(e_ex,ksn_ex,yerr=ksn_u_ex,xerr=e_u_ex,
              linestyle='none',c='grey',zorder=2,elinewidth=0.5) 
plt.xlim((10,10000))
plt.xscale('log')
plt.ylim((0,600))
# plt.ylim(10,10**3)
# plt.yscale('log')
plt.xlabel('Erosion Rate [m/Myr]')
plt.ylabel('$k_{sn}$ [m]')
plt.legend(loc='best')
plt.show()


