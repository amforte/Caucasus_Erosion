#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 10:06:48 2021

@author: aforte
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma
from scipy.stats import weibull_min
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize_scalar
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

def min_tau_c(X,k_e,ks,ksu,e,eu,R,c,s,ns,seed):
    cl=stim.set_constants(R,k_e,dist_type='weibull',tau_c=X)
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

# Load in Data from Erosion Rate basins
edf=pd.read_csv('data_tables/gc_ero_master_table.csv')
ecenters=pd.read_csv('data_tables/grdc_outlines/Ebsns.csv')
erun=pd.read_csv('result_tables/estimate_runoff_power.csv')

ksn=edf['mean_ksn'].to_numpy()
e=edf['St_E_rate_m_Myr'].to_numpy()
eu=edf['St_Ext_Unc'].to_numpy()
ksnu=edf['se_ksn'].to_numpy()
emaxZ=edf['max_el'].to_numpy()/1000
eminZ=edf['outlet_elevation'].to_numpy()/1000
erZ=emaxZ-eminZ
emSN=edf['mean_SNOWstd'].to_numpy()
emR=erun['mean_runoff'].to_numpy()
ex=ecenters['lon'].to_numpy()
ey=ecenters['lat'].to_numpy()

plot_ind=False

# Determine Len
N=len(mR)

### Elbow Plot To Determine Optimal Clusters ####
Xb=np.concatenate((cb.reshape(len(cb),1),mR.reshape(len(mR),1)),axis=1)
Xw=np.concatenate((cw.reshape(len(cw),1),mR.reshape(len(mR),1)),axis=1)

# Scale data
scalerb=StandardScaler().fit(Xb)
XSb=scalerb.transform(Xb)

scalerw=StandardScaler().fit(Xw)
XSw=scalerw.transform(Xw)

# Set random seed for reproducibility
seed=5
num_iterations=500

inertiasb=[]
distortionsb=[]
K_rng=range(1,15)
for i in K_rng:
    kmb=KMeans(n_clusters=i,max_iter=5000,random_state=seed).fit(XSb)
    inertiasb.append(kmb.inertia_)
    distortionsb.append(sum(np.min(cdist(XSb,kmb.cluster_centers_,'euclidean'),axis=1))
                        / XSb.shape[0])
plt.figure(num=1,figsize=(7,4))
ax1=plt.subplot(2,1,1)
ax1.plot(K_rng,inertiasb,'bx-')    
plt.xlabel('Number of Clusters')
plt.ylabel('Intertia')
ax2=plt.subplot(2,1,2)
ax2.plot(K_rng,distortionsb,'bx-')
plt.xlabel('Number of Clusters')
plt.ylabel('Distortion')

### Optimal Cluster Number Based on Elbow ###
num_clustb=4

kmb=KMeans(n_clusters=num_clustb,max_iter=5000,random_state=seed).fit(XSb)

cluster_labels=kmb.labels_
data=np.concatenate((ID.reshape((len(ID),1)),cluster_labels.reshape(len(ID),1)),axis=1)
clustdf=pd.DataFrame(data,columns=['grdc_id','cluster'])
clustdf.to_csv('result_tables/grdc_basin_clusters.csv',index=False)

### Start Plotting   
color_list=['maroon','dodgerblue','darkorange','darkolivegreen','crimson','blue']

### Manually Classify    
eidx=np.ones(emR.shape)*4
eidx[np.logical_and(emR>3.5,emaxZ<2.75)]=0
eidx[np.logical_and(emR>3.5,emaxZ>=2.75)]=3 
eidx[np.logical_and(emR<3.5,emaxZ<3.1)]=1
eidx[np.logical_and(emR<3.5,emaxZ>=3.1)]=2

# eidx=np.ones(emR.shape)*4
# eidx[np.logical_and(emR>3.5,emaxZ<2.5)]=0
# eidx[np.logical_and(emR<3.5,emaxZ<3.1)]=1
# eidx[np.logical_and(emR<3.5,emaxZ>3.5)]=2
# eidx[np.logical_and(emR>3.5,emaxZ>=2.5)]=3 

# eidx=np.ones(emR.shape)*4
# eidx[np.logical_and(emR>3.5,erZ<2)]=0
# eidx[np.logical_and(emR>3.5,erZ>=2)]=3 
# eidx[np.logical_and(emR<3.5,erZ<2.4)]=1
# eidx[np.logical_and(emR<3.5,erZ>=2.4)]=2

plt.figure(num=2,figsize=(25,20))
plt.subplot(2,2,1)
for i in range(num_clustb):
    plt.scatter(mSN[kmb.labels_==i],maxZ[kmb.labels_==i],c=color_list[i])
    plt.scatter(emSN[eidx==i],emaxZ[eidx==i],c=color_list[i],zorder=0,marker='s',s=20,alpha=0.5)
plt.scatter(emSN[eidx==4],emaxZ[eidx==4],s=10,c='gray',zorder=2,marker='s',alpha=0.5)
plt.xlabel('Snow STD')
plt.ylabel('Max Elevation [km]')

plt.subplot(2,2,2)
for i in range(num_clustb):
    plt.scatter(cb[kmb.labels_==i],maxZ[kmb.labels_==i],c=color_list[i])
   
plt.xlabel('Shape')
plt.ylabel('Max Elevation [km]')

plt.subplot(2,2,3)
for i in range(num_clustb):
    plt.scatter(mR[kmb.labels_==i],maxZ[kmb.labels_==i],c=color_list[i])
    plt.scatter(emR[eidx==i],emaxZ[eidx==i],c=color_list[i],zorder=0,marker='s',s=20,alpha=0.5)
plt.scatter(emR[eidx==4],emaxZ[eidx==4],s=10,c='gray',zorder=2,marker='s',alpha=0.5)
plt.xlabel('Mean Runoff [mm/day]')
plt.ylabel('Max Elevation [km]')

plt.subplot(2,2,4)
for i in range(N):
    lldf=pd.read_csv('data_tables/grdc_outlines/Bsn_'+str(ID[i])+'.csv')
    lab=kmb.labels_[i]
    for j in range(num_clustb):
        if lab==j:
            plt.fill(lldf['Lon'],lldf['Lat'],c=color_list[j],zorder=0,alpha=0.5) 
for i in range(num_clustb):
    plt.scatter(ex[eidx==i],ey[eidx==i],c=color_list[i],zorder=0,marker='s',s=10,alpha=0.5)
plt.scatter(ex[eidx==4],ey[eidx==4],s=10,c='gray',zorder=2,marker='s',alpha=0.5)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.gca().set_aspect('equal', adjustable='box')


## 
plt.figure(num=3,figsize=(25,20))
plt.subplot(2,2,1)
for i in range(num_clustb):
    plt.scatter(cb[kmb.labels_==i],mR[kmb.labels_==i],c=color_list[i])
plt.xlabel('Shape')
plt.ylabel('Mean Runoff [mm/day]')

plt.subplot(2,2,2)
for i in range(num_clustb):
    plt.scatter(sb[kmb.labels_==i],mR[kmb.labels_==i],c=color_list[i])
plt.xlabel('Scale')
plt.ylabel('Mean Runoff [mm/day]')

plt.subplot(2,2,3)
for i in range(num_clustb):
    plt.scatter(mR[kmb.labels_==i],maxZ[kmb.labels_==i],c=color_list[i])
    plt.scatter(emR[eidx==i],emaxZ[eidx==i],c=color_list[i],zorder=0,marker='s',s=20,alpha=0.5)
plt.scatter(emR[eidx==4],emaxZ[eidx==4],s=10,c='gray',zorder=2,marker='s',alpha=0.5)
plt.xlabel('Mean Runoff [mm/day]')
plt.ylabel('Max Elevation [km]')

plt.subplot(2,2,4)
for i in range(N):
    lldf=pd.read_csv('data_tables/grdc_outlines/Bsn_'+str(ID[i])+'.csv')
    lab=kmb.labels_[i]
    for j in range(num_clustb):
        if lab==j:
            plt.fill(lldf['Lon'],lldf['Lat'],c=color_list[j],zorder=0,alpha=0.5) 
for i in range(num_clustb):
    plt.scatter(ex[eidx==i],ey[eidx==i],c=color_list[i],zorder=0,marker='s',s=10,alpha=0.5)
plt.scatter(ex[eidx==4],ey[eidx==4],s=10,c='gray',zorder=2,marker='s',alpha=0.5)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.gca().set_aspect('equal', adjustable='box')

## Determine population values
# Empty Arrays
clust_num=np.zeros((num_clustb))
cb_pop=np.zeros((num_clustb))
sb_pop=np.zeros((num_clustb))
mR_pop=np.zeros((num_clustb))
cmb=np.zeros((num_clustb))
smb=np.zeros((num_clustb))

for i in range(num_clustb):
    idx=kmb.labels_==i
    clust_num[i]=i
    cb_pop[i]=np.mean(cb[idx])
    sb_pop[i]=np.mean(sb[idx])
    mR_pop[i]=np.mean(mR[idx])
    ListQs=[]
    ListQf=[]
    for j in range(len(mR[idx])):
        df=pd.read_csv('data_tables/grdc_discharge_time_series/GRDC_'+str(ID[idx][j])+'.csv')
        Q=df['Q'].to_numpy()
        [Qs,Qf]=survive(Q)
        ListQs.append(Qs)
        ListQf.append(Qf)
    Qsaccum=np.concatenate(ListQs,axis=0)
    Qfaccum=np.concatenate(ListQf,axis=0)
    [Qsm,Qfm,QsQ1,QsQ3,QfQ1,QfQ3]=bin_Q(Qsaccum,Qfaccum)
    [cmb[i],smb[i],_,_]=weibull_mt(Qsm,Qfm,mR_pop[i],1.5,1) 
    
### Optimize k_e and tau_c
k_e_optim=np.zeros((len(e)))
tau_c_optim=np.zeros((len(e))) 
for i in range(len(e)):    
    args=(ksn[i],ksnu[i],e[i],eu[i],emR[i],cmb[eidx[i].astype(int)],smb[eidx[i].astype(int)],num_iterations,5)
    res=minimize_scalar(min_k_e,args=args,bounds=[1e-20,1e-6],method='bounded',
                        options={'maxiter':500000,'xatol':1e-20})
    k_e_optim[i]=res.x     
    
k_e_o=np.zeros((num_clustb))
for i in range(num_clustb):
    k_e_o[i]=np.median(k_e_optim[eidx==i])
k_e_fix=np.median(k_e_optim)  

for i in range(len(e)):    
    args=(k_e_fix,ksn[i],ksnu[i],e[i],eu[i],emR[i],cmb[eidx[i].astype(int)],smb[eidx[i].astype(int)],num_iterations,5)
    res=minimize_scalar(min_tau_c,args=args,bounds=[10,90],method='bounded',
                        options={'maxiter':500000,'xatol':1e-20})
    tau_c_optim[i]=res.x

tau_c_o=np.zeros((num_clustb))
for i in range(num_clustb):
    tau_c_o[i]=np.median(tau_c_optim[eidx==i])
tau_c_fix=np.median(tau_c_optim)

### Output 
clustmdata=np.concatenate((clust_num.reshape((len(clust_num),1)),
                     cb_pop.reshape((len(cb_pop),1)),
                     sb_pop.reshape((len(sb_pop),1)),
                     mR_pop.reshape((len(mR_pop),1)),
                     cmb.reshape((len(cmb),1)),
                     smb.reshape((len(smb),1)),
                     k_e_o.reshape((len(k_e_o),1)),
                     tau_c_o.reshape((len(tau_c_o),1))),axis=1)
clustmdf=pd.DataFrame(clustmdata,columns=['cluster','c_mean','s_mean','r_mean','c_aggr','s_aggr','k_e','tau_c'])
clustmdf.to_csv('result_tables/grdc_mean_clusters.csv',index=False)
    
# Fixed
k_e=np.ones((4))*k_e_fix
t_c=np.ones((4))*tau_c_fix

plt.figure(num=4,figsize=(25,10))

for i in range(num_clustb):
    idx=kmb.labels_==i
    
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
        plt.scatter(Qs*mR[idx][j],Qf,s=5,c='gray',alpha=0.5,zorder=0)
        plt.plot(Qs*mR[idx][j],weibull_min.sf(Qs,c=cb[idx][j],loc=0,scale=sb[idx][j]),zorder=1,linewidth=0.5,c=color_list[i],alpha=0.75)
        ListQs.append(Qs)
        ListQf.append(Qf)
    Qsaccum=np.concatenate(ListQs,axis=0)
    Qfaccum=np.concatenate(ListQf,axis=0)
    [Qsm,Qfm,QsQ1,QsQ3,QfQ1,QfQ3]=bin_Q(Qsaccum,Qfaccum)
    
    Qs=np.logspace(-1,2,100)
    plt.plot(Qs*mR_pop[i],weibull_min.sf(Qs,c=cmb[i],loc=0,scale=smb[i]),zorder=2,linewidth=2,c=color_list[i],linestyle='-')
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
    plt.errorbar(e[eidx==i],ksn[eidx==i],ksnu[eidx==i],eu[eidx==i],ecolor=color_list[i],linestyle='',elinewidth=0.5)
    plt.scatter(e[eidx==i],ksn[eidx==i],s=10,marker='s',c=color_list[i],alpha=0.5,zorder=0)
    
    if i==1:
        plt.errorbar(e[eidx==4],ksn[eidx==4],ksnu[eidx==4],eu[eidx==4],ecolor='gray',linestyle='',elinewidth=0.5)
        plt.scatter(e[eidx==4],ksn[eidx==4],s=10,marker='s',c='gray',alpha=0.5,zorder=0) 
    elif i==2:
        plt.errorbar(e[eidx==4],ksn[eidx==4],ksnu[eidx==4],eu[eidx==4],ecolor='gray',linestyle='',elinewidth=0.5)
        plt.scatter(e[eidx==4],ksn[eidx==4],s=10,marker='s',c='gray',alpha=0.5,zorder=0)         
    
    wclb=stim.set_constants(mR_pop[i],k_e[i],dist_type='weibull',tau_c=t_c[i])
    [KSb,Eb,_,_]=stim.stim_range(cmb[i],wcl,sc=smb[i],max_ksn=550)    
    plt.plot(Eb,KSb,c=color_list[i],zorder=2,linewidth=2,linestyle='-',label='Fit to Composite')
    plt.legend(loc='best')   

    plt.xlabel('Erosion Rate [m/Myr]')
    plt.ylabel('$k_{sn}$')
    plt.xlim((10,10000))
    plt.xscale('log')
    plt.ylim((0,550))
    
plt.figure(N+7,figsize=(20,20))
plt.subplot(2,2,1)
for i in range(num_clustb):
    plt.scatter(emR[eidx==i],k_e_optim[eidx==i],c=color_list[i],zorder=1)
    plt.axhline(k_e_o[i],c=color_list[i],linestyle=':')
plt.axhline(np.median(k_e_optim),c='k',linestyle=':')
plt.xlabel('Mean Runoff [mm/day]')
plt.ylabel('Optimized $k_{e}$')
plt.yscale('log')

plt.subplot(2,2,2)
for i in range(num_clustb):
    plt.scatter(emaxZ[eidx==i],k_e_optim[eidx==i],c=color_list[i],zorder=1)
    plt.axhline(k_e_o[i],c=color_list[i],linestyle=':')
plt.axhline(np.median(k_e_optim),c='k',linestyle=':')
plt.xlabel('Max Elevation [km]')
plt.ylabel('Optimized $k_{e}$')
plt.yscale('log')

plt.subplot(2,2,3)
for i in range(num_clustb):
    plt.scatter(emR[eidx==i],tau_c_optim[eidx==i],c=color_list[i],zorder=1)
    plt.axhline(tau_c_o[i],c=color_list[i],linestyle=':')
plt.axhline(np.median(tau_c_optim),c='k',linestyle=':')
plt.xlabel('Mean Runoff [mm/day]')
plt.ylabel(r'Optimized $\tau_c$')

plt.subplot(2,2,4)
for i in range(num_clustb):
    plt.scatter(emaxZ[eidx==i],tau_c_optim[eidx==i],c=color_list[i],zorder=1)
    plt.axhline(tau_c_o[i],c=color_list[i],linestyle=':')
plt.axhline(np.median(tau_c_optim),c='k',linestyle=':')
plt.xlabel('Max Elevation [km]')
plt.ylabel(r'Optimized $\tau_c$')

out_data=np.concatenate((eidx.reshape((len(eidx),1)),k_e_optim.reshape((len(eidx),1)),tau_c_optim.reshape((len(eidx),1))),axis=1)
dfout=pd.DataFrame(out_data,columns=['cluster','k_e','tau_c'])
dfout.to_csv('result_tables/optimized_ero_k_e_tau_c.csv',index=False)
