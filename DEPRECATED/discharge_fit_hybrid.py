#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 15:48:56 2021

@author: aforte
"""
import pandas as pd
import numpy as np
from scipy.stats import weibull_min
from scipy import odr
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import glob


def survive(Q):
    Qstar=Q/np.mean(Q)
    Qstar_sort=np.sort(Qstar)
    Qn=len(Qstar)
    Qrank=np.arange(1,Qn+1,1)
    Q_freq_excd=(Qn+1-Qrank)/Qn
    return Qstar_sort,Q_freq_excd

def extract_interval(a,b,Q,m):
    # Generate month array
    month_array=np.zeros((12,12))
    month_array[0,:]=np.linspace(1,12,12)
    for i in range(11):
        month_array[i+1,:]=np.roll(month_array[i,:],-1)
    month_array=month_array.astype('int')
    # Extract Interval
    interval=a.astype('int')
    month0=b.astype('int')
    ix=month_array[month0,0:interval]
    idx=np.isin(m,ix)
    # Parse record
    Q1=Q[idx]
    Q2=Q[~idx]
    # Calculate survival function
    [Qs1,Qf1]=survive(Q1)
    [Qs2,Qf2]=survive(Q2)
    return Qs1,Qf1,Qs2,Qf2

def find_interval(a,b,Q,m):
    [Qs1,Qf1,Qs2,Qf2]=extract_interval(a,b,Q,m)
    # Fit distribution parts
    if np.any(np.isnan(Qs1)):
        c1=np.nan 
        l1=np.nan 
        s1=np.nan
        res1=np.inf
    else:
        [c1,l1,s1]=weibull_min.fit(Qs1,floc=0,method='MM')
        # Calculate projected values
        Qf1p=weibull_min.logsf(Qs1,c=c1,loc=l1,scale=s1) 
        # Calculate residuals
        res1=np.log(Qf1)-Qf1p         
    if np.any(np.isnan(Qs2)):
        c2=np.nan 
        l2=np.nan 
        s2=np.nan
        res2=np.inf
    else:
        [c2,l2,s2]=weibull_min.fit(Qs2,floc=0,method='MM')
        # Calculate projected values
        Qf2p=weibull_min.logsf(Qs2,c=c2,loc=l2,scale=s2)
        # Calculate residuals
        res2=np.log(Qf2)-Qf2p
    ssr1=np.sum(res1**2)
    ssr2=np.sum(res2**2)
    return ssr1+ssr2,c1,s1,c2,s2

def hybrid_weibull(B,x):
    return B[0]*weibull_min.sf(x,c=B[1],loc=0,scale=B[2]) + (1-B[0])*weibull_min.sf(x,c=B[3],loc=0,scale=B[4])
wbl_model=odr.Model(hybrid_weibull)

def hybrid_weibull_log(B,x):
    return B[0]*weibull_min.logsf(x,c=B[1],loc=0,scale=B[2]) + (1-B[0])*weibull_min.logsf(x,c=B[3],loc=0,scale=B[4])
wbl_log_model=odr.Model(hybrid_weibull_log)

def hybrid_weibull_RMSE(B,x,y):
    yp=B[0]*weibull_min.sf(x,c=B[1],loc=0,scale=B[2]) + (1-B[0])*weibull_min.sf(x,c=B[3],loc=0,scale=B[4])
    return np.sqrt(np.sum(((yp-y)**2))/len(y))

def hybrid_weibull_log_RMSE(B,x,y):
    yp=B[0]*weibull_min.logsf(x,c=B[1],loc=0,scale=B[2]) + (1-B[0])*weibull_min.logsf(x,c=B[3],loc=0,scale=B[4])
    return np.sqrt(np.sum(((yp-y)**2))/len(y))
     
#Build File List    
files=glob.glob('GRDC_discharge/*.csv') 
N=len(files)
tdf=pd.read_csv('result_tables/GRDC_Distribution_Fits.csv')
ct=tdf['c_best'].to_numpy()
st=tdf['s_best'].to_numpy()
IDS=tdf['GRDC_ID'].to_numpy().astype(int)

return_interval=2 # Years
tail_return=(1/(return_interval*365.25))

mn_obs=np.zeros((N))
mn_best=np.zeros((N))
mn_whole=np.zeros((N))
mn_ODRlin=np.zeros((N))
mn_ODRlog=np.zeros((N))
mn_RMSElin=np.zeros((N))
mn_RMSElog=np.zeros((N))

tail_obs=np.zeros((N))
tail_best=np.zeros((N))
tail_whole=np.zeros((N))
tail_ODRlin=np.zeros((N))
tail_ODRlog=np.zeros((N))
tail_RMSElin=np.zeros((N))
tail_RMSElog=np.zeros((N))

for i in range(N):
    # Read Files
    df=pd.read_csv(files[i])
    Q=df['Q'].to_numpy()
    m=df['mnth'].to_numpy()
    R=df['R'].to_numpy()
    mR=np.mean(R)
    mn_obs[i]=mR
    # Calculate exceedance frequency and sorted month index (survival function)
    [Qs,Qf]=survive(Q)
    
    # Generic whole distribution fit for comparison
    [cw,lw,sw]=weibull_min.fit(Qs,floc=0,method='MM')
    
    # Manually minimize to find the best 2 component seasonal block
    int_array=np.arange(1,7,1)
    ssn_array=np.arange(0,12,1)
    r=np.zeros((len(int_array),len(ssn_array)))
    c1=np.zeros((len(int_array),len(ssn_array)))
    s1=np.zeros((len(int_array),len(ssn_array)))
    c2=np.zeros((len(int_array),len(ssn_array)))
    s2=np.zeros((len(int_array),len(ssn_array)))
    for j in range(len(int_array)):
        for k in range(len(ssn_array)):
            [r[j,k],c1[j,k],s1[j,k],c2[j,k],s2[j,k]]=find_interval(int_array[j],ssn_array[k],Q,m)
    ind = np.unravel_index(np.argmin(r, axis=None), r.shape)
    # Find best
    int_best=int_array[ind[0]]
    ssn_best=ssn_array[ind[1]]
    c1s=c1[ind]
    c2s=c2[ind]
    s1s=s1[ind]
    s2s=s2[ind]
    
    # Use minimization to extract implied distributions and best fit parameters
    # for the two blocks
    [Qs1s,Qs2s,Qf1s,Qf2s]=extract_interval(int_best,ssn_best,Q,m)
    
    # Calculate fraction of year occupied by first fraction
    frac1=int_best/12

    # Fit the fractional cdf using results from the minimization as start points
    # Using ODR (unbounded)
    odr_data=odr.RealData(Qs,Qf)
    wbl_obj=odr.ODR(odr_data,wbl_model,beta0=[frac1,c1s,s1s,c2s,s2s])
    wbl_rslt=wbl_obj.run()
    
    odr_log_data=odr.RealData(Qs,np.log(Qf))
    wbl_log_obj=odr.ODR(odr_log_data,wbl_log_model,beta0=[frac1,c1s,s1s,c2s,s2s])
    wbl_log_rslt=wbl_log_obj.run()
    
    # Fit using minizimaiton on RMSE (bounded)
    bnds=((0,1),(0,None),(0,None),(0,None),(0,None))
    X0=[frac1,c1s,s1s,c2s,s2s]
    r=minimize(hybrid_weibull_RMSE,X0,bounds=bnds,args=(Qs,Qf))
    rlog=minimize(hybrid_weibull_log_RMSE,X0,bounds=bnds,args=(Qs,np.log(Qf)))

    # Extract id
    str1=files[i]
    str2=str1.replace('GRDC_discharge/GRDC_','')
    str3=str2.replace('.csv','')
    ID=np.array([str3]).astype('int')
    idx=IDS==ID
    
    mnth_list=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    
    # Plot
    plt.figure(i,figsize=(10,20))
    
    plt.subplot(3,1,1)
    plt.title('Basin '+str3)
    Qstar_array=np.logspace(-2,2,100)
    plt.plot(Qstar_array,weibull_min.sf(Qstar_array,c=c1s,loc=0,scale=s1s),
             c='c',label='Component 1 : F = '+str(np.round(frac1,3))+'; M0 = '+mnth_list[ssn_best]+'; WL = '+str(int_best)+' mnths',zorder=2)
    plt.plot(Qstar_array,weibull_min.sf(Qstar_array,c=c2s,loc=0,scale=s2s),c='g',label='Component 2 : 1-F = '+str(np.round(1-frac1,3)),zorder=2)
    plt.plot(Qstar_array,weibull_min.sf(Qstar_array,c=ct[idx],loc=0,scale=st[idx]),c='k',linestyle='-',label='Tail and Mean Minimization Fit',zorder=3)    
    plt.plot(Qstar_array,weibull_min.sf(Qstar_array,c=cw,loc=lw,scale=sw),c='k',linestyle='--',label='Whole Distribution Fit',zorder=3)
    plt.scatter(Qs,Qf,c='gray',zorder=1,s=50,label='Observed')
    plt.xscale('log')
    plt.yscale('log')
    plt.ylim((10**-4.5,1))
    plt.xlim((10**-2,10**2))
    plt.legend(loc='best')
    
    plt.subplot(3,1,2)
    plt.plot(Qs,hybrid_weibull(wbl_rslt.beta,Qs),c='b',label='Minimize Survival; F = '+str(np.round(wbl_rslt.beta[0],3)),zorder=2)
    plt.plot(Qs,hybrid_weibull(wbl_log_rslt.beta,Qs),c='r',label='Minimize Log Survival; F = '+str(np.round(wbl_log_rslt.beta[0],3)),zorder=2)
    plt.plot(Qstar_array,weibull_min.sf(Qstar_array,c=ct[idx],loc=0,scale=st[idx]),c='k',linestyle='-',label='Tail and Mean Minimization Fit',zorder=3)     
    plt.plot(Qstar_array,weibull_min.sf(Qstar_array,c=cw,loc=lw,scale=sw),c='k',linestyle='--',label='Whole Distribution Fit',zorder=3)    
    plt.scatter(Qs,Qf,c='gray',zorder=1,s=50,label='Observed')   
    plt.xscale('log')
    plt.yscale('log')
    plt.ylim((10**-4.5,1))
    plt.xlim((10**-2,10**2))
    plt.legend(loc='best')
    plt.xlabel('Q*') 
    plt.title('ODR (Unbounded)')
    
    plt.subplot(3,1,3)
    plt.plot(Qs,hybrid_weibull(r.x,Qs),c='b',label='Minimize Survival; F = '+str(np.round(r.x[0],3)),zorder=2)
    plt.plot(Qs,hybrid_weibull(rlog.x,Qs),c='r',label='Minimize Log Survival; F = '+str(np.round(rlog.x[0],3)),zorder=2)
    plt.plot(Qstar_array,weibull_min.sf(Qstar_array,c=ct[idx],loc=0,scale=st[idx]),c='k',linestyle='-',label='Tail and Mean Minimization Fit',zorder=3) 
    plt.plot(Qstar_array,weibull_min.sf(Qstar_array,c=cw,loc=lw,scale=sw),c='k',linestyle='--',label='Whole Distribution Fit',zorder=3)
    plt.scatter(Qs,Qf,c='gray',zorder=1,s=50,label='Observed')   
    plt.xscale('log')
    plt.yscale('log')
    plt.ylim((10**-4.5,1))
    plt.xlim((10**-2,10**2))
    plt.legend(loc='best')
    plt.xlabel('Q*') 
    plt.title('Minimize RMSE (Bounded)')

    # plt.savefig('figures_temp/Basin_'+str3+'.png') 
    
    # Determine mean and tail values
    ODR_lin=hybrid_weibull(wbl_rslt.beta,Qs)
    ODR_log=hybrid_weibull(wbl_log_rslt.beta,Qs)
    RMSE_lin=hybrid_weibull(r.x,Qs)
    RMSE_log=hybrid_weibull(rlog.x,Qs)
    
    ODR_lin_mn=(wbl_rslt.beta[0]*weibull_min.mean(c=wbl_rslt.beta[1],loc=0,scale=wbl_rslt.beta[2]))+((1-wbl_rslt.beta[0])*weibull_min.mean(c=wbl_rslt.beta[3],loc=0,scale=wbl_rslt.beta[4]))
    ODR_log_mn=(wbl_log_rslt.beta[0]*weibull_min.mean(c=wbl_log_rslt.beta[1],loc=0,scale=wbl_log_rslt.beta[2]))+((1-wbl_log_rslt.beta[0])*weibull_min.mean(c=wbl_log_rslt.beta[3],loc=0,scale=wbl_log_rslt.beta[4]))
    RMSE_lin_mn=(r.x[0]*weibull_min.mean(c=r.x[1],loc=0,scale=r.x[2]))+((1-r.x[0])*weibull_min.mean(c=r.x[3],loc=0,scale=r.x[4]))
    RMSE_log_mn=(rlog.x[0]*weibull_min.mean(c=rlog.x[1],loc=0,scale=rlog.x[2]))+((1-rlog.x[0])*weibull_min.mean(c=rlog.x[3],loc=0,scale=rlog.x[4]))
       
    best=weibull_min.mean(c=ct[idx],loc=0,scale=st[idx])
    whole=weibull_min.mean(c=cw,loc=lw,scale=sw)
    
    mn_whole[i]=np.mean(whole)*mR
    mn_best[i]=np.mean(best)*mR
    mn_ODRlin[i]=np.mean(ODR_lin_mn)*mR
    mn_ODRlog[i]=np.mean(ODR_log_mn)*mR
    mn_RMSElin[i]=np.mean(RMSE_lin_mn)*mR
    mn_RMSElog[i]=np.mean(RMSE_log_mn)*mR

    tail_obs[i]=Qs[np.argmin(np.abs(Qf-tail_return))]*mR
    tail_whole[i]=weibull_min.isf(tail_return,cw,loc=0,scale=sw)*mR
    tail_best[i]=weibull_min.isf(tail_return,ct[idx],loc=0,scale=st[idx])*mR    
    tail_ODRlin[i]=Qs[np.argmin(np.abs(ODR_lin-tail_return))]*mR
    tail_ODRlog[i]=Qs[np.argmin(np.abs(ODR_log-tail_return))]*mR
    tail_RMSElin[i]=Qs[np.argmin(np.abs(RMSE_lin-tail_return))]*mR
    tail_RMSElog[i]=Qs[np.argmin(np.abs(RMSE_log-tail_return))]*mR   
    

plt.figure(N+1,figsize=(15,15))
plt.subplot(2,2,1)
plt.plot(np.linspace(0,7),np.linspace(0,7),linestyle=':',c='k')
plt.scatter(mn_obs,mn_whole,s=20,c='gray',label='Whole')
plt.scatter(mn_obs,mn_best,s=20,c='k',label='Mean + Tail')
plt.scatter(mn_obs,mn_ODRlin,s=20,c='b',label='ODR Survival')
plt.scatter(mn_obs,mn_ODRlog,s=20,c='r',label='ODR Log Survival') 
plt.xlabel('Observed Mean Runoff')
plt.ylabel('Implied Mean Runoff')
plt.legend(loc='best')
plt.xlim((0,7))
plt.ylim((0,7))
   
plt.subplot(2,2,2)
plt.plot(np.linspace(0,7),np.linspace(0,7),linestyle=':',c='k')
plt.scatter(mn_obs,mn_whole,s=20,c='gray',label='Whole')
plt.scatter(mn_obs,mn_best,s=20,c='k',label='Mean + Tail')
plt.scatter(mn_obs,mn_RMSElin,s=20,c='b',label='RMSE Survival')
plt.scatter(mn_obs,mn_RMSElog,s=20,c='r',label='RMSE Log Survival') 
plt.xlabel('Observed Mean Runoff')
plt.ylabel('Implied Mean Runoff') 
plt.legend(loc='best') 
plt.xlim((0,7))
plt.ylim((0,7))  

plt.subplot(2,2,3)
plt.plot(np.linspace(0,85),np.linspace(0,85),linestyle=':',c='k')
plt.scatter(tail_obs,tail_whole,s=20,c='gray',label='Whole')
plt.scatter(tail_obs,tail_best,s=20,c='k',label='Mean + Tail')
plt.scatter(tail_obs,tail_ODRlin,s=20,c='b',label='ODR Survival')
plt.scatter(tail_obs,tail_ODRlog,s=20,c='r',label='ODR Log Survival') 
plt.xlabel('Observed 2 Year Runoff')
plt.ylabel('Implied 2 Year Runoff')
plt.legend(loc='best')
plt.xlim((0,85))
plt.ylim((0,85))

plt.subplot(2,2,4)
plt.plot(np.linspace(0,85),np.linspace(0,85),linestyle=':',c='k')
plt.scatter(tail_obs,tail_whole,s=20,c='gray',label='Whole')
plt.scatter(tail_obs,tail_best,s=20,c='k',label='Mean + Tail')
plt.scatter(tail_obs,tail_RMSElin,s=20,c='b',label='RMSE Survival')
plt.scatter(tail_obs,tail_RMSElog,s=20,c='r',label='RMSE Log Survival') 
plt.xlabel('Observed 2 Year Runoff')
plt.ylabel('Implied 2 Year Runoff')
plt.legend(loc='best')   
plt.xlim((0,85))
plt.ylim((0,85))

# plt.savefig('figures_temp/R_R.png')



