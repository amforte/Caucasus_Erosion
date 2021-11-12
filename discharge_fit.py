#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 16:23:20 2021

@author: amforte
"""

import pandas as pd
import numpy as np
from scipy.stats import weibull_min
from scipy.special import gamma
import matplotlib.pyplot as plt
import glob
from astropy.utils import NumpyRNGContext
from astropy.stats import bootstrap


def weibull_tail_fit(x,thresh):
    # x is sorted
    n=len(x)
    rank=np.arange(1,n+1,1)
    y=(n+1-rank)/n
    ix=np.nonzero(y<thresh)[0][:1][0]
    # ix=np.argmin(np.abs(y-thresh))
    xtrim=x[ix:n]
    ytrim=y[ix:n]
    N=len(xtrim)
    xts=np.log(xtrim)
    yts=np.log(-np.log(ytrim))       
    [lin,r,rnk,sng,V]=np.polyfit(xts,yts,1,full=True)
    c=lin[0]
    s=np.exp(-1*lin[1]/c)
    mn=s*gamma(1+(1/c))
    # Convert sum of squared residuals to mean of sum of squared residuals
    res=r/N
    return c,s,mn,N,res

def weibull_tail_bootstrap(x,thresh,test_c,test_s,num_replicates,alpha):
    # Non-parametric bootstrap (i.e. resampling of residuals)
    # x is sorted
    n=len(x)
    rank=np.arange(1,n+1,1)
    y=(n+1-rank)/n
    ix=np.nonzero(y<thresh)[0][:1][0]
    # ix=np.argmin(np.abs(y-thresh))
    xtrim=x[ix:n]
    ytrim=y[ix:n]
    xts=np.log(xtrim)
    yts=np.log(-np.log(ytrim))
    # Calculate indiviudal residuals
    epi=yts-(test_c*xts+(-np.log(test_s)*test_c))
    # Set seed for reproducibility
    bsc=np.zeros((num_replicates))
    bss=np.zeros((num_replicates))    
    with NumpyRNGContext(0):
        epi_rand=bootstrap(epi,num_replicates)
        for i in range(num_replicates):
            lin=np.polyfit(xts,yts+epi_rand[i,:],1)
            c=lin[0]
            s=np.exp(-1*lin[1]/c)
            bsc[i]=c
            bss[i]=s
    # Sort and determine confidence intervals based provided alpha
    l=(1-alpha)/2
    u=alpha+((1-alpha)/2)
    c_l=np.percentile(bsc,l*100)
    c_u=np.percentile(bsc,u*100)
    s_l=np.percentile(bss,l*100)
    s_u=np.percentile(bss,u*100)
    c_std=np.std(bsc)
    s_std=np.std(bss)
    return c_l,c_u,s_l,s_u,c_std,s_std   

# Build File List    
files=glob.glob('GRDC_discharge/*.csv') 
# Define threshold array
thresh_array=np.linspace(0.01,0.6,50)
nt=len(thresh_array) 
# Count and generate empty arrays
N=len(files)
mnR=np.zeros((N,1))
whole_fits=np.zeros((N,3))
ct=np.zeros((N,nt))
st=np.zeros((N,nt))
mnt=np.zeros((N,nt))
Nt=np.zeros((N,nt))
res=np.zeros((N,nt))
Nta=np.zeros((N,nt))
RRt=np.zeros((N,nt))

ct_best=np.zeros((N,1))
ct_best_ci=np.zeros((N,2))
ct_best_std=np.zeros((N,1))
st_best=np.zeros((N,1))
st_best_ci=np.zeros((N,2))
st_best_std=np.zeros((N,1))

mnt_best=np.zeros((N,1))
Nta_best=np.zeros((N,1))
thresh_best=np.zeros((N,1))
IDs=np.zeros((N,1))

# Generate Return Flood Targets
return_interval=2 # Years
targ_return=(1/(return_interval*365.25))
targ_obs=np.zeros((N,1))
targ_best=np.zeros((N,1))
targ_whole=np.zeros((N,1))

# Flag for individual plots
plot_ind=False

# Flag for tail optimization method
# 'MSS' - minimization based on mean sum of squares of linear fit on natural log transformed tail
# 'RETURN' - minimization based on difference between observed and measured return flood specified by targ_return
tail_method='MSS'

# Set weights for mean and tail components of fit
# Larger values indicate this component will be weighted more heavily in the minimization
tail_weight=1
runoff_weight=1.5

# Begin File Loop
for i in range(N):
    # Read Files
    df=pd.read_csv(files[i])
    # Extract, normalize, and sort
    Q=df['Q'].to_numpy()
    R=df['R'].to_numpy()
    yr=df['yr'].to_numpy()
    yrspan=np.max(yr)-np.min(yr)
    Qstar=Q/np.mean(Q)
    Qstar_sort=np.sort(Qstar)
    # Begin thresh loop
    for j in range(nt):    
        # Fit tail
        try:
            [ct[i,j],st[i,j],mnt[i,j],Nt[i,j],res[i,j]]=weibull_tail_fit(Qstar_sort,thresh_array[j])
            Nta[i,j]=Nt[i,j]/yrspan
            RRt[i,j]=weibull_min.isf(targ_return,ct[i,j],loc=0,scale=st[i,j])*np.mean(R)
        except:
            # This except block catches thresholds above which zeros are included
            # in the tail fit, which are undefined in ln-ln space
            ct[i,j]=np.NAN
            st[i,j]=np.NAN
            mnt[i,j]=np.NAN
            Nt[i,j]=np.NAN
            res[i,j]=np.NAN
            Nta[i,j]=np.NAN
            RRt[i,j]=np.NAN
    # Fit whole distribution
    [c,l,s]=weibull_min.fit(Qstar_sort,floc=0,method='MM')
    # Calculate exceedance frequency
    Qn=len(Qstar)
    Qrank=np.arange(1,Qn+1,1)
    Q_freq_excd=(Qn+1-Qrank)/Qn  
    # Package output
    mnR[i,0]=np.mean(R)
    whole_fits[i,0]=c
    whole_fits[i,1]=s
    whole_fits[i,2]=weibull_min.mean(c,loc=0,scale=s)
    # Find observed runoff at target return interval
    fix1=np.argmin(np.abs(Q_freq_excd-targ_return))
    targ_obs[i,0]=Qstar_sort[fix1]*np.mean(R)    
    # Find local minimum
    impR=np.mean(R)*mnt[i,:]
    difR=np.abs(impR-np.mean(R))
    runoff_min=(difR/np.nanmax(difR))*runoff_weight
    if tail_method=='MSS':
        tail_min=(res[i,:]/np.nanmax(res[i,:]))*tail_weight
    elif tail_method=='RETURN':
        difRR=np.abs(targ_obs[i,0]-RRt[i,:])
        tail_min=(difRR/np.nanmax(difRR))*tail_weight
    lm=tail_min+runoff_min
    ix=np.nanargmin(lm)
    # Store the minimum values
    ct_best[i,0]=ct[i,ix]
    st_best[i,0]=st[i,ix]
    mnt_best[i,0]=mnt[i,ix]
    Nta_best[i,0]=Nt[i,ix]/yrspan
    thresh_best[i,0]=thresh_array[ix]
    # Determine boostrap confidence intervals
    ct_best_ci[i,0],ct_best_ci[i,1],st_best_ci[i,0],st_best_ci[i,1],ct_best_std[i,0],st_best_std[i,0]=weibull_tail_bootstrap(Qstar_sort,thresh_best[i,0],ct_best[i,0],st_best[i,0],1000,0.95)
    # Calculate runoff at target return interval
    targ_whole[i,0]=weibull_min.isf(targ_return,c,loc=l,scale=s)*np.mean(R)
    targ_best[i,0]=weibull_min.isf(targ_return,ct[i,ix],loc=0,scale=st[i,ix])*np.mean(R)
    # Extract id
    str1=files[i]
    str2=str1.replace('GRDC_discharge/GRDC_','')
    str3=str2.replace('.csv','')
    IDs[i,0]=np.array(str3).astype(int)    
    # Start Plot
    if plot_ind:
        plt.figure(i+1,figsize=(20,10)) 
        # Mean Runoff
        plt.subplot(3,2,1)
        plt.title('Basin '+str3+'; Mean R = '+str(np.round(np.mean(R),1))+'; R2yr = '+str(np.round(targ_obs[i,0],1)))
        plt.axhline(np.mean(R),c='r',linestyle='--',label='Observed Mean Runoff')
        plt.axhline(np.mean(R)*whole_fits[i,2],c='k',linestyle='-',label='Whole Fit Implied Runoff')
        for j in range(nt):
            plt.scatter(Nta[i,j],np.mean(R)*mnt[i,j],c='k',s=20)
        plt.scatter(Nta[i,ix],np.mean(R)*mnt[i,ix],c='b',s=60,zorder=3,label='Minimum')
        plt.ylabel('Runoff [mm/day]')
        plt.legend(loc='best')
        plt.xlim((0,250))
        # Shape Parameter
        plt.subplot(3,2,3)
        plt.axhline(whole_fits[i,0],c='k',linestyle='-',label='Whole Fit c')
        for j in range(nt):
            plt.scatter(Nta[i,j],ct[i,j],c='k',s=20)
        plt.scatter(Nta[i,ix],ct[i,ix],c='b',s=60,zorder=3,label='Minimum')     
        plt.ylabel('Shape Parameter')
        plt.legend(loc='best')
        plt.xlim((0,250))
        # Tail Statistics
        plt.subplot(3,2,5)
        if tail_method=='MSS':
            for j in range(nt):
                plt.scatter(Nta[i,j],res[i,j],c='k',s=20)
            plt.scatter(Nta[i,ix],res[i,ix],c='b',s=60,zorder=3)    
            plt.xlabel('Events Per Year')
            plt.ylabel('Tail MSS')
            plt.xlim((0,250))
        elif tail_method=='RETURN':
            plt.axhline(targ_obs[i,0],c='r',linestyle='--',
                        label='Observed '+str(return_interval)+' Year Runoff')
            plt.axhline(targ_whole[i,0],c='k',linestyle='-',
                       label='Whole Fit '+str(return_interval)+' Year Runoff')
            for j in range(nt):
                plt.scatter(Nta[i,j],RRt[i,j],c='k',s=20)
            plt.scatter(Nta[i,ix],RRt[i,ix],c='b',s=60,zorder=3,label='Minimum')    
            plt.xlabel('Events Per Year')
            plt.ylabel('Tail RMSE')
            plt.xlim((0,250))  
            plt.legend(loc='best')
        # Exceedance Frequency (Survival Function)
        plt.subplot(1,2,2)
        if tail_method=='RETURN':
            plt.axhline(targ_return,c='k',linestyle=':',label='Return Flood Used For Tail Minimization')
        plt.scatter(Qstar_sort,Q_freq_excd,label='Observations',c='r')
        plt.plot(Qstar_sort,weibull_min.sf(Qstar_sort,c,loc=l,scale=s),label='Whole Fit',c='k')
        plt.plot(Qstar_sort,weibull_min.sf(Qstar_sort,ct[i,ix],loc=0,scale=st[i,ix]),label='Minimum',c='b')
        plt.legend(loc='best')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Q*')
        plt.ylim((10**-4,1))
        plt.xlim((10**-2,10**2))

        
# Comparison Plot
plt.figure(N+1,figsize=(10,10))
plt.subplot(2,2,1)
plt.plot(np.arange(0,8,1),np.arange(0,8,1),c='k',linestyle=':')
plt.scatter(mnR,mnt_best*mnR,s=20,c='b',label='Minimum')
plt.scatter(mnR,whole_fits[:,2]*np.ravel(mnR),s=20,c='k',label='Whole Fit')
plt.xlabel('Observed Mean Runoff [mm/day]')
plt.ylabel('Implied Mean Runoff [mm/day]') 
plt.legend(loc='best')
plt.xlim((0,7))
plt.ylim((0,7))
plt.title('Runoff Weight = '+str(runoff_weight))
#
plt.subplot(2,2,2)
plt.plot(np.arange(0,85,5),np.arange(0,85,5),c='k',linestyle=':')
plt.scatter(targ_obs,targ_best,s=20,c='b',label='Minimum')
plt.scatter(targ_obs,targ_whole,s=20,c='k',label='Whole Fit')
plt.xlabel('Observed 2 yr Return Runoff [mm/day]')
plt.ylabel('Implied 2 yr Return Runoff [mm/day]')
plt.legend(loc='best')
plt.title('Tail Weight = '+str(tail_weight))
#
plt.subplot(2,2,3)
st_error=np.transpose(st_best_ci)
st_error[0,:]=np.transpose(st_best)-st_error[0,:]
st_error[1,:]=st_error[1,:]-np.transpose(st_best)
ct_error=np.transpose(ct_best_ci)
ct_error[0,:]=np.transpose(ct_best)-ct_error[0,:]
ct_error[1,:]=np.transpose(ct_best)-ct_error[1,:]
plt.errorbar(st_best,ct_best,yerr=np.ravel(ct_best_std),xerr=np.ravel(st_best_std),ecolor='b',linestyle='',elinewidth=0.5)
plt.scatter(st_best,ct_best,c='b',s=20)
plt.scatter(whole_fits[:,1],whole_fits[:,0],c='k',s=20)
plt.ylabel('Shape Parameter')
plt.xlabel('Scale Parameter')
plt.xlim((0,1.5))
plt.ylim((0,5))
plt.legend(loc='best')
#
plt.subplot(2,2,4)
plt.scatter(ct_best,Nta_best,c='b',s=20)
plt.xlabel('Shape (c)')
plt.ylabel('Threshold (Events/Year)')

# Package output
# Sort arrays by GRDC ID to match other tables
idx=np.argsort(IDs,axis=0)
data=np.concatenate((np.take_along_axis(IDs,idx,0),
                     np.take_along_axis(ct_best,idx,0),
                     np.take_along_axis(ct_best_ci,idx,0),
                     np.take_along_axis(ct_best_std,idx,0),
                     np.take_along_axis(st_best,idx,0),
                     np.take_along_axis(st_best_ci,idx,0),
                     np.take_along_axis(st_best_std,idx,0),
                     np.take_along_axis(mnt_best,idx,0),
                     np.take_along_axis(thresh_best,idx,0),
                     np.take_along_axis(Nta_best,idx,0),
                     np.take_along_axis(whole_fits,idx,0),
                     np.take_along_axis(mnR,idx,0),
                     np.take_along_axis(targ_obs,idx,0),
                     np.take_along_axis(targ_best,idx,0),
                     np.take_along_axis(targ_whole,idx,0)),axis=1)
dfout=pd.DataFrame(data,columns=['GRDC_ID','c_best','c_lo95','c_hi95','c_std',
                                 's_best','s_lo95','s_hi95','s_std',
                                 'mn_best','thresh_best','Npera_best','c_whole',
                                 's_whole','mn_whole','mean_R_obs',
                                 'return2_R_obs','return2_R_best',
                                 'return2_R_whole'])
dfout.to_csv('result_tables/GRDC_Distribution_Fits.csv',index=False)


