#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 14:53:49 2021

@author: aforte
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma
from matplotlib import cm
from astropy.utils import NumpyRNGContext

import stochastic_threshold as stim

def survive(Q):
    Qstar=Q/np.mean(Q)
    Qstar_sort=np.sort(Qstar)
    Qn=len(Qstar)
    Qrank=np.arange(1,Qn+1,1)
    Q_freq_excd=(Qn+1-Qrank)/Qn
    return Qstar_sort,Q_freq_excd

def bin_Q(Qs,Qf):
    bins=np.linspace(10**-4,1,1500)
    ix=np.digitize(Qf,bins)
    d=np.concatenate((ix.reshape((len(ix),1)),Qs.reshape((len(ix),1)),Qf.reshape((len(ix),1))),axis=1)
    df=pd.DataFrame(d,columns=['ix','Qs','Qf'])
    m=df.groupby('ix').mean()
    q1=df.groupby('ix').quantile(0.25)
    q3=df.groupby('ix').quantile(0.75)
    Qsm=m['Qs'].to_numpy()
    Qfm=m['Qf'].to_numpy()
    QsQ1=q1['Qs'].to_numpy()
    QsQ3=q3['Qs'].to_numpy()
    QfQ1=q1['Qf'].to_numpy()
    QfQ3=q3['Qf'].to_numpy()
    Qsm=np.flip(Qsm)
    Qfm=np.flip(Qfm)
    QsQ1=np.flip(QsQ1)
    QsQ3=np.flip(QsQ3)
    QfQ1=np.flip(QfQ1)
    QfQ3=np.flip(QfQ3) 
    return Qsm,Qfm,QsQ1,QsQ3,QfQ1,QfQ3

def min_k_e(X,ks,ksu,e,eu,R,c,s,ns,seed):
    cl=stim.set_constants(R,X,dist_type='weibull')
    ks_dist=np.random.normal(ks,ksu,ns)
    e_dist=np.random.normal(e,eu,ns)
    ep=np.zeros((ns))
    with NumpyRNGContext(seed):
        for i in range(ns):
            [ep[i],_,_]=stim.stim_one(ks_dist[i],c,cl,sc=s)
    rmse=np.sqrt(np.sum((e_dist-ep)**2)/ns)
    return rmse

        
def weibull_tail_fit(x,y,thresh):
    n=len(x)
    ix=np.nonzero(y<thresh)[0][:1][0]
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

def weibull_mt(Qs,Qf,mnR,mean_weight,tail_weight):
    thresh_array=np.linspace(0.01,0.6,50)
    nt=len(thresh_array)
    ct=np.zeros((nt))
    st=np.zeros((nt))
    mnt=np.zeros((nt))
    Nt=np.zeros((nt))
    res=np.zeros((nt))
     
    for i in range(nt):
        try:
            [ct[i],st[i],mnt[i],Nt[i],res[i]]=weibull_tail_fit(Qs,Qf,thresh_array[i])
        except:
            # This except block catches thresholds above which zeros are included
            # in the tail fit, which are undefined in ln-ln space
            ct[i]=np.NAN
            st[i]=np.NAN
            mnt[i]=np.NAN
            Nt[i]=np.NAN
            res[i]=np.NAN
    # Find local minimum
    impR=mnR*mnt
    difR=np.abs(impR-mnR)
    runoff_min=(difR/np.nanmax(difR))*mean_weight
    tail_min=(res/np.nanmax(res))*tail_weight    
    lm=tail_min+runoff_min
    ix=np.nanargmin(lm)
    # Store the minimum values
    ct_best=ct[ix]
    st_best=st[ix]
    mnt_best=mnt[ix]
    thresh_best=thresh_array[ix]
    return ct_best,st_best,mnt_best,thresh_best
    
# Load in data from GRDC basins
df=pd.read_csv('result_tables/GRDC_Distribution_Fits.csv')
gdf=pd.read_csv('data_tables/grdc_summary_values.csv')
cb=df['c_best'].to_numpy()
sb=df['s_best'].to_numpy()
cw=df['c_whole'].to_numpy()
sw=df['s_whole'].to_numpy()
k99=gdf['k_99'].to_numpy()
mR=gdf['R_mm_dy'].to_numpy()
ID=gdf['ID'].to_numpy()

# Filter out the two super low runoff basins
idx=mR>0.3
cb=cb[idx]
sb=sb[idx]
cw=cw[idx]
sw=sw[idx]
k99=k99[idx]
mR=mR[idx]
ID=ID[idx]

plot_ind=False

# Determine Len
N=len(mR)

plt.figure(N+1,figsize=(20,15))
k_e=2.4e-12

cmap=cm.jet_r(np.linspace(0,1,50))
rmap=np.linspace(np.min(mR),np.max(mR),len(cmap))

for i in range(N):
    wcl=stim.set_constants(mR[i],k_e,dist_type='weibull')
    igcl=stim.set_constants(mR[i],k_e)
    
    [Ks,E,E_err,Qsc]=stim.stim_range(k99[i],igcl,max_ksn=550)
    [wKs1,wE1,wE_err1,wQsc1]=stim.stim_range(cb[i],wcl,sc=sb[i],max_ksn=550)
    [wKs2,wE2,wE_err2,wQsc2]=stim.stim_range(cw[i],wcl,sc=sw[i],max_ksn=550)
    
    ix=np.argmin(np.abs(mR[i]-rmap))
    c0=cmap[ix,:]
    
    if plot_ind==True:
        plt.figure(i)
        plt.plot(wE1,wKs1,c=c0,label='Weibull MT')
        plt.plot(wE2,wKs2,c=c0,linestyle=':',label='Weibull Whole')
        plt.plot(E,Ks,c=c0,linestyle='--',label='Inverse Gamma Tail')
        plt.xlabel('E [m/Myr]')
        plt.ylabel('$k_{sn}$ [m]')
        plt.title('Basin = '+str(ID[i])+'; R = '+str(np.round(mR[i],2)))  
        plt.xlim((0,8000))
        plt.legend(loc='best')
    
    if mR[i]>4:
        plt.figure(N+1)
        plt.subplot(1,2,1)
        plt.plot(((E-wE1)),Ks,zorder=2,c=c0)
        
        plt.figure(N+1)
        plt.subplot(1,2,2)
        plt.plot(((wE1-wE2)),wKs1,zorder=2,c=c0)
    else:
        plt.figure(N+1)
        plt.subplot(1,2,1)
        plt.plot(((E-wE1)),Ks,zorder=2,c=c0,linestyle=':')
        
        plt.figure(N+1)
        plt.subplot(1,2,2)
        plt.plot(((wE1-wE2)),wKs1,zorder=2,c=c0,linestyle=':') 

plt.subplot(1,2,1)
plt.xlabel('Inv Gamma - Weibull MT [m/Myr]')
plt.ylabel('$k_{sn}$ [m]')
plt.xlim((-9000,2500))

plt.subplot(1,2,2)
plt.xlabel('Weibull MT - Weibull Whole [m/Myr]' )
plt.ylabel('$k_{sn}$ [m]')
plt.xlim((-9000,2500))