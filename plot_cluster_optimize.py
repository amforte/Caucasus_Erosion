#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 14 14:17:09 2021

@author: amforte
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import weibull_min

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

# Load in data from GRDC basins
df=pd.read_csv('result_tables/GRDC_Distribution_Fits.csv')
mR=df['mean_R_obs'].to_numpy()
gdf=pd.read_csv('data_tables/grdc_summary_values.csv')
cb=df['c_best'].to_numpy()
sb=df['s_best'].to_numpy()
cw=df['c_whole'].to_numpy()
sw=df['s_whole'].to_numpy()
mSN=gdf['ssnstd'].to_numpy()
maxZ=gdf['maxz'].to_numpy()/1000
minZ=gdf['minz'].to_numpy()/1000
mnZ=gdf['mnz'].to_numpy()/1000
rZ=maxZ-minZ
ID=gdf['ID'].to_numpy()

# Determine Len
NGRDC=len(mR)

# Load in Data from Erosion Rate basins
edf=pd.read_csv('data_tables/gc_ero_master_table.csv')
ecenters=pd.read_csv('data_tables/grdc_outlines/Ebsns.csv')
erun=pd.read_csv('result_tables/estimate_runoff_power.csv')
edist=pd.read_csv('data_tables/swath_distances.txt')

ksn=edf['mean_ksn'].to_numpy()
e=edf['St_E_rate_m_Myr'].to_numpy()
eu=edf['St_Ext_Unc'].to_numpy()
ksnu=edf['se_ksn'].to_numpy()
emaxZ=edf['max_el'].to_numpy()/1000
eminZ=edf['outlet_elevation'].to_numpy()/1000
emnZ=edf['mean_el'].to_numpy()/1000
erZ=emaxZ-eminZ
emSN=edf['mean_SNOWstd'].to_numpy()
emR=erun['mean_runoff'].to_numpy()
ex=ecenters['lon'].to_numpy()
ey=ecenters['lat'].to_numpy()
ealong=edist['D_along_SW'].to_numpy()
eacross=edist['D_from_SW'].to_numpy()

edf2=pd.read_csv('result_tables/optimized_ero_k_e_tau_c.csv')
ecluster_label=edf2['cluster'].to_numpy().astype(int)
num_clustb=np.max(ecluster_label)+1
k_e_optim=edf2['k_e_median'].to_numpy()
tau_c_optim=edf2['tau_c_median'].to_numpy()
k_e_q25=edf2['k_e_q25'].to_numpy()
k_e_q75=edf2['k_e_q75'].to_numpy()
tau_c_q25=edf2['tau_c_q25'].to_numpy()
tau_c_q75=edf2['tau_c_q75'].to_numpy()

# Fixed
k_e_fix=np.median(k_e_optim)
tau_c_fix=np.median(tau_c_optim)
k_e=np.ones((4))*k_e_fix
t_c=np.ones((4))*tau_c_fix

# Load in results of clustering of GRDC basins
clustdf=pd.read_csv('result_tables/grdc_basin_clusters.csv')
ID=clustdf['grdc_id'].to_numpy().astype(int)
cluster_label=clustdf['cluster'].to_numpy().astype(int)

# Load in cluster population info
clustmdf=pd.read_csv('result_tables/grdc_mean_clusters.csv')
cb_pop=clustmdf['c_mean'].to_numpy()
sb_pop=clustmdf['s_mean'].to_numpy()
mR_pop=clustmdf['r_mean'].to_numpy()
cmb=clustmdf['c_aggr'].to_numpy()
smb=clustmdf['s_aggr'].to_numpy()
k_e_o=clustmdf['k_e'].to_numpy()
tau_c_o=clustmdf['tau_c'].to_numpy()

### Start Plotting   
color_list=['maroon','dodgerblue','darkorange','darkolivegreen','crimson','blue']

plt.figure(num=2,figsize=(25,20))
plt.subplot(2,2,1)
for i in range(num_clustb):
    plt.scatter(mSN[cluster_label==i],maxZ[cluster_label==i],c=color_list[i])
    plt.scatter(emSN[ecluster_label==i],emaxZ[ecluster_label==i],c=color_list[i],zorder=0,marker='s',s=20,alpha=0.5)
plt.scatter(emSN[ecluster_label==4],emaxZ[ecluster_label==4],s=10,c='gray',zorder=2,marker='s',alpha=0.5)
plt.xlabel('Snow STD')
plt.ylabel('Max Elevation [km]')

plt.subplot(2,2,2)
for i in range(num_clustb):
    plt.scatter(cb[cluster_label==i],maxZ[cluster_label==i],c=color_list[i])
   
plt.xlabel('Shape')
plt.ylabel('Max Elevation [km]')

plt.subplot(2,2,3)
for i in range(num_clustb):
    plt.scatter(mR[cluster_label==i],maxZ[cluster_label==i],c=color_list[i])
    plt.scatter(emR[ecluster_label==i],emaxZ[ecluster_label==i],c=color_list[i],zorder=0,marker='s',s=20,alpha=0.5)
plt.scatter(emR[ecluster_label==4],emaxZ[ecluster_label==4],s=10,c='gray',zorder=2,marker='s',alpha=0.5)
plt.xlabel('Mean Runoff [mm/day]')
plt.ylabel('Max Elevation [km]')

plt.subplot(2,2,4)
for i in range(NGRDC):
    lldf=pd.read_csv('data_tables/grdc_outlines/Bsn_'+str(ID[i])+'.csv')
    lab=cluster_label[i]
    for j in range(num_clustb):
        if lab==j:
            plt.fill(lldf['Lon'],lldf['Lat'],c=color_list[j],zorder=0,alpha=0.5) 
for i in range(num_clustb):
    plt.scatter(ex[ecluster_label==i],ey[ecluster_label==i],c=color_list[i],zorder=0,marker='s',s=10,alpha=0.5)
plt.scatter(ex[ecluster_label==4],ey[ecluster_label==4],s=10,c='gray',zorder=2,marker='s',alpha=0.5)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.gca().set_aspect('equal', adjustable='box')

## 
f2=plt.figure(num=3,figsize=(20,20))
plt.subplot(2,2,3)
for i in range(num_clustb):
    plt.scatter(cb[cluster_label==i],mR[cluster_label==i],c=color_list[i])
    plt.scatter(cmb[i],mR_pop[i],c=color_list[i],s=60,marker='s',edgecolors='k')
plt.xlabel('Shape')
plt.ylabel('Mean Runoff [mm/day]')

plt.subplot(2,2,4)
for i in range(num_clustb):
    plt.scatter(sb[cluster_label==i],mR[cluster_label==i],c=color_list[i])
    plt.scatter(smb[i],mR_pop[i],c=color_list[i],s=60,marker='s',edgecolors='k')
plt.xlabel('Scale')
plt.ylabel('Mean Runoff [mm/day]')

plt.subplot(2,2,1)
for i in range(num_clustb):
    # plt.scatter(mR[cluster_label==i],mnZ[cluster_label==i],c=color_list[i])
    # plt.scatter(emR[ecluster_label==i],emnZ[ecluster_label==i],c=color_list[i],zorder=0,marker='s',s=20,alpha=0.5)    
    plt.scatter(mR[cluster_label==i],maxZ[cluster_label==i],c=color_list[i])
    plt.scatter(emR[ecluster_label==i],emaxZ[ecluster_label==i],c=color_list[i],zorder=0,marker='D',s=20,alpha=0.5)
plt.scatter(emR[ecluster_label==4],emaxZ[ecluster_label==4],s=10,c='gray',zorder=2,marker='D',alpha=0.5)
plt.xlabel('Mean Runoff [mm/day]')
plt.ylabel('Max Elevation [km]')

plt.subplot(2,2,2)
for i in range(NGRDC):
    lldf=pd.read_csv('data_tables/grdc_outlines/Bsn_'+str(ID[i])+'.csv')
    lab=cluster_label[i]
    for j in range(num_clustb):
        if lab==j:
            plt.fill(lldf['Lon'],lldf['Lat'],c=color_list[j],zorder=0,alpha=0.5) 
for i in range(num_clustb):
    plt.scatter(ex[ecluster_label==i],ey[ecluster_label==i],c=color_list[i],zorder=0,marker='s',s=10,alpha=0.5)
plt.scatter(ex[ecluster_label==4],ey[ecluster_label==4],s=10,c='gray',zorder=2,marker='s',alpha=0.5)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.gca().set_aspect('equal', adjustable='box')


f3=plt.figure(num=4,figsize=(25,10))
for i in range(num_clustb):
    idx=cluster_label==i
    
    plt.subplot(2,4,i+1)
    
    plt.title('Cluster '+str(i+1)) 
    Qs=np.logspace(-1,2,100)
    plt.plot(Qs*mR_pop[i],weibull_min.sf(Qs,c=cb_pop[i],loc=0,scale=sb_pop[i]),zorder=2,linewidth=2,c=color_list[i],linestyle='--')
    ListQs=[]
    ListQf=[]
    for j in range(len(mR[idx])):
        df=pd.read_csv('data_tables/grdc_discharge_time_series/GRDC_'+str(ID[idx][j])+'.csv')
        Q=df['Q'].to_numpy()
        [Qs,Qf]=survive(Q)
        # plt.scatter(Qs*mR[idx][j],Qf,s=5,c='gray',alpha=0.5,zorder=0)
        plt.plot(Qs*mR[idx][j],weibull_min.sf(Qs,c=cb[idx][j],loc=0,scale=sb[idx][j]),zorder=1,linewidth=0.5,c=color_list[i],alpha=0.75)
        ListQs.append(Qs)
        ListQf.append(Qf)
    Qsaccum=np.concatenate(ListQs,axis=0)
    Qfaccum=np.concatenate(ListQf,axis=0)
    [Qsm,Qfm,QsQ1,QsQ3,QfQ1,QfQ3]=bin_Q(Qsaccum,Qfaccum)
    
    Qs=np.logspace(-1,2,100)
    plt.plot(Qs*mR_pop[i],weibull_min.sf(Qs,c=cmb[i],loc=0,scale=smb[i]),zorder=2,linewidth=2,c=color_list[i],linestyle='-')
    # plt.scatter(Qsm*mR_pop[i],Qfm,c='k',s=10,zorder=1)
    
    QsIQ=np.concatenate((QsQ1.reshape(len(QsQ1),1),QsQ3.reshape(len(QsQ3),1)),axis=1)
    Qfm_=np.concatenate((Qfm.reshape(len(Qfm),1),Qfm.reshape(len(Qfm),1)),axis=1)
    # plt.plot(np.transpose(QsIQ)*mR_pop[i],np.transpose(Qfm_),c='k',linewidth=0.5,zorder=0)
    
    plt.yscale('log')
    plt.ylim((10**-4,1))
    plt.xlabel('Runoff [mm/day]')
    plt.ylabel('Exceedance Frequency')
    plt.xlim((0,125))

    plt.subplot(2,4,i+5)
    
    wcl=stim.set_constants(mR_pop[i],k_e[i],dist_type='weibull',tau_c=t_c[i])
    [KSpop,Epop,_,_]=stim.stim_range(cb_pop[i],wcl,sc=sb_pop[i],max_ksn=550)
    plt.plot(Epop,KSpop,c=color_list[i],zorder=2,linewidth=2,label='Mean of Cluster',linestyle='--')
    for j in range(len(mR[idx])):
        wcl0=stim.set_constants(mR[idx][j],k_e[i],dist_type='weibull',tau_c=t_c[i])
        [KS0,E0,_,_]=stim.stim_range(cb[idx][j],wcl0,sc=sb[idx][j],max_ksn=550)
        plt.plot(E0,KS0,c=color_list[i],zorder=1,linewidth=0.5,alpha=0.75)
    plt.errorbar(e[ecluster_label==i],ksn[ecluster_label==i],ksnu[ecluster_label==i],eu[ecluster_label==i],ecolor=color_list[i],linestyle='',elinewidth=0.5)
    plt.scatter(e[ecluster_label==i],ksn[ecluster_label==i],s=10,marker='s',c=color_list[i],alpha=0.5,zorder=0)
    
    if i==1:
        plt.errorbar(e[ecluster_label==4],ksn[ecluster_label==4],ksnu[ecluster_label==4],eu[ecluster_label==4],ecolor='gray',linestyle='',elinewidth=0.5)
        plt.scatter(e[ecluster_label==4],ksn[ecluster_label==4],s=10,marker='s',c='gray',alpha=0.5,zorder=0) 
    elif i==2:
        plt.errorbar(e[ecluster_label==4],ksn[ecluster_label==4],ksnu[ecluster_label==4],eu[ecluster_label==4],ecolor='gray',linestyle='',elinewidth=0.5)
        plt.scatter(e[ecluster_label==4],ksn[ecluster_label==4],s=10,marker='s',c='gray',alpha=0.5,zorder=0)         
    
    wclb=stim.set_constants(mR_pop[i],k_e[i],dist_type='weibull',tau_c=t_c[i])
    [KSb,Eb,_,_]=stim.stim_range(cmb[i],wcl,sc=smb[i],max_ksn=550)    
    plt.plot(Eb,KSb,c=color_list[i],zorder=2,linewidth=2,linestyle='-',label='Fit to Composite')
    plt.legend(loc='best')   

    plt.xlabel('Erosion Rate [m/Myr]')
    plt.ylabel('$k_{sn}$')
    plt.xlim((10,10000))
    plt.xscale('log')
    plt.ylim((0,550))


f3a=plt.figure(num=6,figsize=(25,10))
for i in range(num_clustb):
    idx=cluster_label==i
    
    plt.subplot(2,4,i+1)
    
    plt.title('Cluster '+str(i+1)) 
    Qs=np.logspace(-1,2,100)
    # plt.plot(Qs*mR_pop[i],weibull_min.sf(Qs,c=cb_pop[i],loc=0,scale=sb_pop[i]),zorder=2,linewidth=2,c=color_list[i],linestyle='--')
    ListQs=[]
    ListQf=[]
    for j in range(len(mR[idx])):
        df=pd.read_csv('data_tables/grdc_discharge_time_series/GRDC_'+str(ID[idx][j])+'.csv')
        Q=df['Q'].to_numpy()
        [Qs,Qf]=survive(Q)
        plt.scatter(Qs*mR[idx][j],Qf,s=5,c='gray',alpha=0.5,zorder=0,edgecolors=None)
        # plt.plot(Qs*mR[idx][j],weibull_min.sf(Qs,c=cb[idx][j],loc=0,scale=sb[idx][j]),zorder=1,linewidth=0.5,c=color_list[i],alpha=0.75)
        ListQs.append(Qs)
        ListQf.append(Qf)
    Qsaccum=np.concatenate(ListQs,axis=0)
    Qfaccum=np.concatenate(ListQf,axis=0)
    [Qsm,Qfm,QsQ1,QsQ3,QfQ1,QfQ3]=bin_Q(Qsaccum,Qfaccum)
    
    Qs=np.logspace(-1,2,100)
    # plt.plot(Qs*mR_pop[i],weibull_min.sf(Qs,c=cmb[i],loc=0,scale=smb[i]),zorder=2,linewidth=2,c=color_list[i],linestyle='-')
    plt.scatter(Qsm*mR_pop[i],Qfm,c='k',s=10,zorder=1)
    
    QsIQ=np.concatenate((QsQ1.reshape(len(QsQ1),1),QsQ3.reshape(len(QsQ3),1)),axis=1)
    Qfm_=np.concatenate((Qfm.reshape(len(Qfm),1),Qfm.reshape(len(Qfm),1)),axis=1)
    plt.plot(np.transpose(QsIQ)*mR_pop[i],np.transpose(Qfm_),c='k',linewidth=0.5,zorder=0)
    
    plt.yscale('log')
    plt.ylim((10**-4,1))
    plt.xlabel('Runoff [mm/day]')
    plt.ylabel('Exceedance Frequency')
    plt.xlim((0,125))

    plt.subplot(2,4,i+5)
    
    wcl=stim.set_constants(mR_pop[i],k_e[i],dist_type='weibull',tau_c=t_c[i])
    [KSpop,Epop,_,_]=stim.stim_range(cb_pop[i],wcl,sc=sb_pop[i],max_ksn=550)
    plt.plot(Epop,KSpop,c=color_list[i],zorder=2,linewidth=2,label='Mean of Cluster',linestyle='--')
    for j in range(len(mR[idx])):
        wcl0=stim.set_constants(mR[idx][j],k_e[i],dist_type='weibull',tau_c=t_c[i])
        [KS0,E0,_,_]=stim.stim_range(cb[idx][j],wcl0,sc=sb[idx][j],max_ksn=550)
        plt.plot(E0,KS0,c=color_list[i],zorder=1,linewidth=0.5,alpha=0.75)
    plt.errorbar(e[ecluster_label==i],ksn[ecluster_label==i],ksnu[ecluster_label==i],eu[ecluster_label==i],ecolor=color_list[i],linestyle='',elinewidth=0.5)
    plt.scatter(e[ecluster_label==i],ksn[ecluster_label==i],s=10,marker='s',c=color_list[i],alpha=0.5,zorder=0)
    
    if i==1:
        plt.errorbar(e[ecluster_label==4],ksn[ecluster_label==4],ksnu[ecluster_label==4],eu[ecluster_label==4],ecolor='gray',linestyle='',elinewidth=0.5)
        plt.scatter(e[ecluster_label==4],ksn[ecluster_label==4],s=10,marker='s',c='gray',alpha=0.5,zorder=0) 
    elif i==2:
        plt.errorbar(e[ecluster_label==4],ksn[ecluster_label==4],ksnu[ecluster_label==4],eu[ecluster_label==4],ecolor='gray',linestyle='',elinewidth=0.5)
        plt.scatter(e[ecluster_label==4],ksn[ecluster_label==4],s=10,marker='s',c='gray',alpha=0.5,zorder=0)         
    
    wclb=stim.set_constants(mR_pop[i],k_e[i],dist_type='weibull',tau_c=t_c[i])
    [KSb,Eb,_,_]=stim.stim_range(cmb[i],wcl,sc=smb[i],max_ksn=550)    
    plt.plot(Eb,KSb,c=color_list[i],zorder=2,linewidth=2,linestyle='-',label='Fit to Composite')
    plt.legend(loc='best')   

    plt.xlabel('Erosion Rate [m/Myr]')
    plt.ylabel('$k_{sn}$')
    plt.xlim((10,10000))
    plt.xscale('log')
    plt.ylim((0,550))

## Evaluate optimizations

# Structure error bars
k_e_l=k_e_optim-k_e_q25
k_e_u=k_e_q75-k_e_optim
k_e_err=np.concatenate((k_e_l.reshape(1,len(k_e_l)),k_e_u.reshape((1,len(k_e_u)))),axis=0)
t_c_l=tau_c_optim-tau_c_q25
t_c_u=tau_c_q75-tau_c_optim
t_c_err=np.concatenate((t_c_l.reshape(1,len(t_c_l)),t_c_u.reshape((1,len(t_c_u)))),axis=0) 
   
f4=plt.figure(num=5,figsize=(30,15))
plt.subplot(2,3,1)
for i in range(num_clustb):
    plt.errorbar(emR[ecluster_label==i],k_e_optim[ecluster_label==i],yerr=k_e_err[:,ecluster_label==i],ecolor=color_list[i],zorder=0,linestyle='')
    plt.scatter(emR[ecluster_label==i],k_e_optim[ecluster_label==i],c=color_list[i],zorder=1)
    plt.axhline(k_e_o[i],c=color_list[i],linestyle=':')
plt.axhline(np.median(k_e_optim),c='k',linestyle=':')
plt.xlabel('Mean Runoff [mm/day]')
plt.ylabel('Optimized $k_{e}$')
plt.yscale('log')

plt.subplot(2,3,2)
for i in range(num_clustb):
    plt.errorbar(emaxZ[ecluster_label==i],k_e_optim[ecluster_label==i],yerr=k_e_err[:,ecluster_label==i],ecolor=color_list[i],zorder=0,linestyle='')
    plt.scatter(emaxZ[ecluster_label==i],k_e_optim[ecluster_label==i],c=color_list[i],zorder=1)
    plt.axhline(k_e_o[i],c=color_list[i],linestyle=':')
plt.axhline(np.median(k_e_optim),c='k',linestyle=':')
plt.xlabel('Max Elevation [km]')
plt.ylabel('Optimized $k_{e}$')
plt.yscale('log')

plt.subplot(2,3,3)
for i in range(num_clustb):
    plt.errorbar(eacross[ecluster_label==i],k_e_optim[ecluster_label==i],yerr=k_e_err[:,ecluster_label==i],ecolor=color_list[i],zorder=0,linestyle='')
    plt.scatter(eacross[ecluster_label==i],k_e_optim[ecluster_label==i],c=color_list[i],zorder=1)
    plt.axhline(k_e_o[i],c=color_list[i],linestyle=':')
plt.axhline(np.median(k_e_optim),c='k',linestyle=':')
plt.xlabel('Distance Across Swath [km]')
plt.ylabel('Optimized $k_{e}$')
plt.yscale('log')

ax3=plt.subplot(2,3,4)
for i in range(num_clustb):
    plt.errorbar(emR[ecluster_label==i],tau_c_optim[ecluster_label==i],yerr=t_c_err[:,ecluster_label==i],ecolor=color_list[i],zorder=0,linestyle='')
    plt.scatter(emR[ecluster_label==i],tau_c_optim[ecluster_label==i],c=color_list[i],zorder=1)
    plt.axhline(tau_c_o[i],c=color_list[i],linestyle=':')
plt.axhline(np.median(tau_c_optim),c='k',linestyle=':')
plt.xlabel('Mean Runoff [mm/day]')
plt.ylabel(r'Optimized $\tau_c$')
plt.ylim((20,90))
plt.yscale('log')

ax3_2=ax3.twinx()
ax3_2.set_ylabel(r'Optimized $\Psi_c$')
ax3_2.set_yscale('log')
low_psi=(k_e_fix)*(20**1.5)
hi_psi=(k_e_fix)*(90**1.5)
ax3_2.set_ylim((low_psi,hi_psi))
ax3_2.set_ylabel(r'Optimized $\Psi_c$')

ax4=plt.subplot(2,3,5)
for i in range(num_clustb):
    plt.errorbar(emaxZ[ecluster_label==i],tau_c_optim[ecluster_label==i],yerr=t_c_err[:,ecluster_label==i],ecolor=color_list[i],zorder=0,linestyle='')
    plt.scatter(emaxZ[ecluster_label==i],tau_c_optim[ecluster_label==i],c=color_list[i],zorder=1)
    plt.axhline(tau_c_o[i],c=color_list[i],linestyle=':')
plt.axhline(np.median(tau_c_optim),c='k',linestyle=':')
plt.xlabel('Max Elevation [km]')
plt.ylabel(r'Optimized $\tau_c$')
plt.ylim((20,90))
plt.yscale('log')

ax4_2=ax4.twinx()
ax4_2.set_ylabel(r'Optimized $\Psi_c$')
ax4_2.set_yscale('log')
low_psi=(k_e_fix)*(20**1.5)
hi_psi=(k_e_fix)*(90**1.5)
ax4_2.set_ylim((low_psi,hi_psi))
ax4_2.set_ylabel(r'Optimized $\Psi_c$')


ax5=plt.subplot(2,3,6)
for i in range(num_clustb):
    plt.errorbar(eacross[ecluster_label==i],tau_c_optim[ecluster_label==i],yerr=t_c_err[:,ecluster_label==i],ecolor=color_list[i],zorder=0,linestyle='')
    plt.scatter(eacross[ecluster_label==i],tau_c_optim[ecluster_label==i],c=color_list[i],zorder=1)
    plt.axhline(tau_c_o[i],c=color_list[i],linestyle=':')
plt.axhline(np.median(tau_c_optim),c='k',linestyle=':')
plt.xlabel('Distance Across Swath [km]')
plt.ylabel(r'Optimized $\tau_c$')
plt.ylim((20,90))
plt.yscale('log')

ax5_2=ax5.twinx()
ax5_2.set_ylabel(r'Optimized $\Psi_c$')
ax5_2.set_yscale('log')
low_psi=(k_e_fix)*(20**1.5)
hi_psi=(k_e_fix)*(90**1.5)
ax5_2.set_ylim((low_psi,hi_psi))
ax5_2.set_ylabel(r'Optimized $\Psi_c$')

######
f4a=plt.figure(num=10,figsize=(15,5))

# Multiplicative factors for scaling bubbles
mf=20
imf=1/2

ax1=plt.subplot(1,2,1)
for i in range(num_clustb):
    plt.errorbar(eacross[ecluster_label==i],k_e_optim[ecluster_label==i],
                 yerr=k_e_err[:,ecluster_label==i],ecolor=color_list[i],
                 zorder=0,linestyle='')
    plt.scatter(eacross[ecluster_label==i],k_e_optim[ecluster_label==i],
                c=color_list[i],zorder=1,s=np.log10(e[ecluster_label==i]*imf)*mf)
    plt.axhline(k_e_o[i],c=color_list[i],linestyle=':')

plt.scatter(60,1e-8,s=np.log10(10*imf)*mf,c='gray')
plt.scatter(65,1e-8,s=np.log10(100*imf)*mf,c='gray')
plt.scatter(70,1e-8,s=np.log10(1000*imf)*mf,c='gray')
plt.scatter(75,1e-8,s=np.log10(10000*imf)*mf,c='gray')

plt.axhline(np.median(k_e_optim),c='k',linestyle=':')
plt.xlabel('Distance Across Swath [km]')
plt.ylabel('Optimized $k_{e}$')
plt.yscale('log')
plt.ylim((10**-13,10**-7))

ax1r=ax1.twinx()
ax1r.set_ylim(((10**-13)*(tau_c_fix**1.5),(10**-7)*(tau_c_fix**1.5)))
ax1r.set_yscale('log')
ax1r.set_ylabel(r'Optimized $\Psi_c$')

ax2=plt.subplot(1,2,2)
plt.xlabel('Distance Across Swath [km]')
low_psi=(k_e_fix)*(20**1.5)
hi_psi=(k_e_fix)*(90**1.5)
plt.ylim((low_psi,hi_psi))
plt.yscale('log')

ax2r=ax2.twinx()
ax2r.set_yscale('log')
ax2r.set_ylim((20,90))
ax2r.set_ylabel(r'Optimized $\tau_c$')
for i in range(num_clustb):
    ax2r.errorbar(eacross[ecluster_label==i],
                  tau_c_optim[ecluster_label==i],yerr=t_c_err[:,ecluster_label==i],
                  ecolor=color_list[i],zorder=0,linestyle='')
    ax2r.scatter(eacross[ecluster_label==i],tau_c_optim[ecluster_label==i],
                 c=color_list[i],zorder=1,s=np.log10(e[ecluster_label==i]*imf)*mf)
    ax2r.axhline(tau_c_o[i],c=color_list[i],linestyle=':')
ax2r.axhline(np.median(tau_c_optim),c='k',linestyle=':')

# f2.savefig('cluster_class.pdf')
# f3.savefig('cluster_ksn_e.pdf')
# f3a.savefig('cluster_ksn_e.tif',dpi=300)
# f4.savefig('cluster_ke.pdf')
# f4a.savefig('cluster_ke2.pdf')