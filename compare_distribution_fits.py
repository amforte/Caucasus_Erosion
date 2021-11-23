#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compares results of different methods of fitting the runoff distributions, both
in terms of how they fit the runoff distributions, but also the implications for
the projected ksn-E relationship using the optimized k_e from the analysis.

Written by Adam M. Forte for 
"Low variability runoff inhibits coupling of climate, tectonics, and 
topography in the Greater Caucasus"

If you use this code or derivatives, please cite the original paper.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.stats import weibull_min
from scipy.stats import invgamma

import stochastic_threshold as stim

def survive(Q):
    Qstar=Q/np.mean(Q)
    Qstar_sort=np.sort(Qstar)
    Qn=len(Qstar)
    Qrank=np.arange(1,Qn+1,1)
    Q_freq_excd=(Qn+1-Qrank)/Qn
    return Qstar_sort,Q_freq_excd
 
# Load in data from GRDC basins
df=pd.read_csv('result_tables/GRDC_Distribution_Fits.csv')
# Weibull distribution parameters
cb=df['c_best'].to_numpy()
sb=df['s_best'].to_numpy()
cblo=df['c_lo95'].to_numpy()
cbhi=df['c_hi95'].to_numpy()
sblo=df['s_lo95'].to_numpy()
sbhi=df['s_hi95'].to_numpy()
cw=df['c_whole'].to_numpy()
sw=df['s_whole'].to_numpy()
mb=df['mn_best'].to_numpy()
mw=df['mn_whole'].to_numpy()

# Weibull 2yr Values
R2b=df['return2_R_best'].to_numpy()
R2w=df['return2_R_whole'].to_numpy()
# Inverse gamma distribution parameters
k99=df['k_tail99'].to_numpy()
klog=df['k_whole_log'].to_numpy()
klin=df['k_whole_lin'].to_numpy()
# Inverse 2yr Values
R2k99=df['return2_R_k_tail99'].to_numpy()
R2klin=df['return2_R_k_whole_lin'].to_numpy()
R2klog=df['return2_R_k_whole_log'].to_numpy()
# General info
mR=df['mean_R_obs'].to_numpy()
R2o=df['return2_R_obs'].to_numpy()
ID=df['GRDC_ID'].to_numpy().astype(int)

plot_ind=True

# Determine Len
N=len(mR)

k_e=1e-11

cmap=cm.jet_r(np.linspace(0,1,50))
rmap=np.linspace(np.min(mR),np.max(mR),len(cmap))

plt.figure(N+1,figsize=(20,15))

for i in range(N):
    wcl=stim.set_constants(mR[i],k_e,dist_type='weibull')
    igcl=stim.set_constants(mR[i],k_e)
    
    [kKs1,kE1,_]=stim.stim_range(k99[i],igcl,max_ksn=550)
    [kKs2,kE2,_]=stim.stim_range(klog[i],igcl,max_ksn=550)
    [kKs3,kE3,_]=stim.stim_range(klin[i],igcl,max_ksn=550)
    [wKs1,wE1,_]=stim.stim_range(cb[i],wcl,sc=sb[i],max_ksn=550)
    [wKs2,wE2,_]=stim.stim_range(cw[i],wcl,sc=sw[i],max_ksn=550)
    
    ix=np.argmin(np.abs(mR[i]-rmap))
    c0=cmap[ix,:]
    
    if plot_ind==True:
        plt.figure(i,figsize=(20,10))
        plt.subplot(1,2,1)
        plt.plot(wE1,wKs1,c=c0,label='Weibull MT')
        plt.plot(wE2,wKs2,c=c0,linestyle='--',label='Weibull Whole')
        plt.plot(kE1,kKs1,c=c0,linestyle=':',label='Inverse Gamma Tail')
        plt.plot(kE2,kKs2,c=c0,linestyle=(0,(3,1,1,1)),label='Inverse Gamma Log Whole')
        plt.plot(kE3,kKs3,c=c0,linestyle=(0,(3,3,1,3)),label='Inverse Gamma Lin Whole')
        plt.xlabel('E [m/Myr]')
        plt.ylabel('$k_{sn}$ [m]')
        plt.title('Basin = '+str(ID[i])+'; R = '+str(np.round(mR[i],2)))  
        plt.xlim((0,8000))
        plt.legend(loc='best')
        
        df=pd.read_csv('data_tables/grdc_discharge_time_series/GRDC_'+str(ID[i])+'.csv')
        Q=df['Q'].to_numpy()
        [Qs,Qf]=survive(Q)
        plt.subplot(1,2,2)
        plt.scatter(Qs*mR[i],Qf,s=20,c='gray',zorder=1)
        plt.plot(Qs*mR[i],weibull_min.sf(Qs,cb[i],loc=0,scale=sb[i]),c=c0,label='Weibull MT')
        plt.plot(Qs*mR[i],weibull_min.sf(Qs,cw[i],loc=0,scale=sw[i]),c=c0,linestyle='--',label='Weibull Whole')
        plt.plot(Qs*mR[i],invgamma.sf(Qs,k99[i],loc=0,scale=k99[i]+1),c=c0,linestyle=':',label='Inverse Gamma Tail')
        plt.plot(Qs*mR[i],invgamma.sf(Qs,klog[i],loc=0,scale=klog[i]+1),c=c0,linestyle=(0,(3,1,1,1)),label='Inverse Gamma Log Whole')
        plt.plot(Qs*mR[i],invgamma.sf(Qs,klin[i],loc=0,scale=klin[i]+1),c=c0,linestyle=(0,(3,3,1,3)),label='Inverse Gamma Lin Whole')        
        plt.yscale('log')
        plt.ylim((10**-4,1))
        plt.ylabel('Exceedance Frequency')
        plt.xlim((0,np.max(Qs)*mR[i]+10))
        plt.xlabel('Runoff [mm/day]')
    
    if mR[i]>4:
        plt.figure(N+1)
        plt.subplot(1,2,1)
        plt.plot(((kE1-wE1)),kKs1,zorder=2,c=c0)
        
        plt.figure(N+1)
        plt.subplot(1,2,2)
        plt.plot(((wE1-wE2)),wKs1,zorder=2,c=c0)
    else:
        plt.figure(N+1)
        plt.subplot(1,2,1)
        plt.plot(((kE1-wE1)),kKs1,zorder=2,c=c0,linestyle=':')
        
        plt.figure(N+1)
        plt.subplot(1,2,2)
        plt.plot(((wE1-wE2)),wKs1,zorder=2,c=c0,linestyle=':') 
        
plt.subplot(1,2,1)
plt.xlabel('Inv Gamma - Weibull MT [m/Myr]')
plt.ylabel('$k_{sn}$ [m]')
plt.xlim((-5000,2500))

plt.subplot(1,2,2)
plt.xlabel('Weibull MT - Weibull Whole [m/Myr]' )
plt.ylabel('$k_{sn}$ [m]')
plt.xlim((-5000,2500))

f1=plt.figure(N+2,figsize=(20,12))

plt.subplot(2,3,1)
plt.plot(np.arange(0,8,1),np.arange(0,8,1),c='k',linestyle=':')
plt.scatter(mR,mb*mR,c='b',s=30,label='Mean and Tail Minimization Fit')
plt.scatter(mR,mw*mR,c='k',s=30,label='Whole Distribution Fit',marker='s')
plt.xlabel('Observed Mean Runoff [mm/day]')
plt.ylabel('Implied Mean Runoff [mm/day]') 
plt.legend(loc='best')
plt.xlim((0,7))
plt.ylim((0,7))

plt.subplot(2,3,2)
plt.plot(np.arange(0,100,5),np.arange(0,100,5),c='k',linestyle=':')
plt.scatter(R2o,R2b,c='b',s=30,label='Mean and Tail Minimization Fit')
plt.scatter(R2o,R2w,c='k',s=30,label='Whole Distribution Fit',marker='s')
plt.xlabel('Observed 2 yr Return Runoff [mm/day]')
plt.ylabel('Implied 2 yr Return Runoff [mm/day]')
plt.xlim((0,85))
plt.ylim((0,85))

plt.subplot(2,3,3)
plt.scatter(sb,cb,c='b',s=30)
plt.scatter(sw,cw,c='k',s=30,marker='s')
plt.ylabel('Shape Parameter')
plt.xlabel('Scale Parameter')

rep_ksn1=200
rep_ksn2=300
rep_ksn3=400

plt.subplot(2,3,4)
IDt=6990100
i=np.argmin(np.abs(ID-IDt))
wcl=stim.set_constants(mR[i],k_e,dist_type='weibull')
[_,Qscb1]=stim.stim_one(rep_ksn1,cb[i],wcl,sc=sb[i])
[_,Qscw1]=stim.stim_one(rep_ksn1,cw[i],wcl,sc=sw[i])
[_,Qscb2]=stim.stim_one(rep_ksn2,cb[i],wcl,sc=sb[i])
[_,Qscw2]=stim.stim_one(rep_ksn2,cw[i],wcl,sc=sw[i])
[_,Qscb3]=stim.stim_one(rep_ksn3,cb[i],wcl,sc=sb[i])
[_,Qscw3]=stim.stim_one(rep_ksn3,cw[i],wcl,sc=sw[i])
df=pd.read_csv('data_tables/grdc_discharge_time_series/GRDC_'+str(ID[i])+'.csv')
Q=df['Q'].to_numpy()
[Qs,Qf]=survive(Q)
plt.scatter(Qs*mR[i],Qf,s=20,c='gray',zorder=1)
plt.plot(Qs*mR[i],weibull_min.sf(Qs,cb[i],loc=0,scale=sb[i]),c='b',label='Mean and Tail Minimization Fit')
plt.plot(Qs*mR[i],weibull_min.sf(Qs,cw[i],loc=0,scale=sw[i]),c='k',linestyle='--',label='Whole Distribution Fit')  
plt.axvline(Qscb1*mR[i],c='g',linestyle='--',label='Critical Runoff $k_{sn}$=200 [m]')
plt.axvline(Qscb2*mR[i],c='orange',linestyle='--',label='Critical Runoff $k_{sn}$=300 [m]')
plt.axvline(Qscb3*mR[i],c='r',linestyle='--',label='Critical Runoff $k_{sn}$=400 [m]')
plt.yscale('log')
plt.ylim((10**-4,1))
plt.ylabel('Exceedance Frequency')
plt.xlim((0,np.max(Qs)*mR[i]+10))
plt.xlabel('Runoff [mm/day]')
plt.title('Basin = '+str(ID[i])+'; R = '+str(np.round(mR[i],2)))  
plt.legend(loc='best')

plt.subplot(2,3,5)
IDt=6983800
i=np.argmin(np.abs(ID-IDt))
wcl=stim.set_constants(mR[i],k_e,dist_type='weibull')
[_,Qscb1]=stim.stim_one(rep_ksn1,cb[i],wcl,sc=sb[i])
[_,Qscw1]=stim.stim_one(rep_ksn1,cw[i],wcl,sc=sw[i])
[_,Qscb2]=stim.stim_one(rep_ksn2,cb[i],wcl,sc=sb[i])
[_,Qscw2]=stim.stim_one(rep_ksn2,cw[i],wcl,sc=sw[i])
[_,Qscb3]=stim.stim_one(rep_ksn3,cb[i],wcl,sc=sb[i])
[_,Qscw3]=stim.stim_one(rep_ksn3,cw[i],wcl,sc=sw[i])
df=pd.read_csv('data_tables/grdc_discharge_time_series/GRDC_'+str(ID[i])+'.csv')
Q=df['Q'].to_numpy()
[Qs,Qf]=survive(Q)
plt.scatter(Qs*mR[i],Qf,s=20,c='gray',zorder=1)
plt.plot(Qs*mR[i],weibull_min.sf(Qs,cb[i],loc=0,scale=sb[i]),c='b',label='Mean and Tail Minimization Fit')
plt.plot(Qs*mR[i],weibull_min.sf(Qs,cw[i],loc=0,scale=sw[i]),c='k',linestyle='--',label='Whole Distribution Fit')  
plt.axvline(Qscb1*mR[i],c='g',linestyle='--',label='Critical Runoff $k_{sn}$=200 [m]')
plt.axvline(Qscb2*mR[i],c='orange',linestyle='--',label='Critical Runoff $k_{sn}$=300 [m]')
plt.axvline(Qscb3*mR[i],c='r',linestyle='--',label='Critical Runoff $k_{sn}$=400 [m]')
plt.yscale('log')
plt.ylim((10**-4,1))
plt.ylabel('Exceedance Frequency')
plt.xlim((0,np.max(Qs)*mR[i]+10))
plt.xlabel('Runoff [mm/day]')
plt.title('Basin = '+str(ID[i])+'; R = '+str(np.round(mR[i],2)))  

plt.subplot(2,3,6)
IDt=6985350
i=np.argmin(np.abs(ID-IDt))
wcl=stim.set_constants(mR[i],k_e,dist_type='weibull')
[_,Qscb1]=stim.stim_one(rep_ksn1,cb[i],wcl,sc=sb[i])
[_,Qscw1]=stim.stim_one(rep_ksn1,cw[i],wcl,sc=sw[i])
[_,Qscb2]=stim.stim_one(rep_ksn2,cb[i],wcl,sc=sb[i])
[_,Qscw2]=stim.stim_one(rep_ksn2,cw[i],wcl,sc=sw[i])
[_,Qscb3]=stim.stim_one(rep_ksn3,cb[i],wcl,sc=sb[i])
[_,Qscw3]=stim.stim_one(rep_ksn3,cw[i],wcl,sc=sw[i])
df=pd.read_csv('data_tables/grdc_discharge_time_series/GRDC_'+str(ID[i])+'.csv')
Q=df['Q'].to_numpy()
[Qs,Qf]=survive(Q)
plt.scatter(Qs*mR[i],Qf,s=20,c='gray',zorder=1)
plt.plot(Qs*mR[i],weibull_min.sf(Qs,cb[i],loc=0,scale=sb[i]),c='b',label='Mean and Tail Minimization Fit')
plt.plot(Qs*mR[i],weibull_min.sf(Qs,cw[i],loc=0,scale=sw[i]),c='k',linestyle='--',label='Whole Distribution Fit')  
plt.axvline(Qscb1*mR[i],c='g',linestyle='--',label='Critical Runoff $k_{sn}$=200 [m]')
plt.axvline(Qscb2*mR[i],c='orange',linestyle='--',label='Critical Runoff $k_{sn}$=300 [m]')
plt.axvline(Qscb3*mR[i],c='r',linestyle='--',label='Critical Runoff $k_{sn}$=400 [m]')
plt.yscale('log')
plt.ylim((10**-4,1))
plt.ylabel('Exceedance Frequency')
plt.xlim((0,np.max(Qs)*mR[i]+10))
plt.xlabel('Runoff [mm/day]')
plt.title('Basin = '+str(ID[i])+'; R = '+str(np.round(mR[i],2)))  

f1.savefig('Compare_MT_Whole_Fit.pdf')
