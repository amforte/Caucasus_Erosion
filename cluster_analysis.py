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
from matplotlib import colors
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
from scipy.stats import skew
from scipy import odr



import stochastic_threshold as stim

# Set random seed for reproducibility
seed=5

# Read in data
df=pd.read_csv('data_tables/gc_ero_master_table.csv')
# Extract main variables of interest
mR=df['mean_runoff'].to_numpy()
mP=df['corrected_mean_trmm'].to_numpy()
RR=df['runoff_ratio'].to_numpy()
mS=df['mean_SNOWstd'].to_numpy()
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
k_e_simple=dfke['k_e_simple'].to_numpy()
est_err=dfke['rmse_bootstrap'].to_numpy()
e_predicted=dfke['E_pred'].to_numpy() 


# Assemble stats (these do not sum to 100, mixed and matched)
perc_carb=(df['rock_type_perc_limestone_minor_clastics'].to_numpy()+
          df['rock_type_perc_limestone_marl'].to_numpy()+
          df['rock_type_perc_limestone'].to_numpy()+
          df['rock_type_perc_marl'].to_numpy())
perc_clastic=(df['rock_type_perc_sandstone_shale'].to_numpy()+
              df['rock_type_perc_sandstone_minor_carbonate'].to_numpy()+
              df['rock_type_perc_shale'].to_numpy()+
              df['rock_type_perc_conglomerates'].to_numpy())
perc_mixed=df['rock_type_perc_limestone_sandstone'].to_numpy()
perc_igmet=(df['rock_type_perc_phyllite_schist'].to_numpy()+
            df['rock_type_perc_volcanic_undiff'].to_numpy()+
            df['rock_type_perc_granite'].to_numpy())
perc_fine=(df['rock_type_perc_marl'].to_numpy()+
            df['rock_type_perc_shale'].to_numpy())
perc_ss=df['rock_type_perc_sandstone_shale'].to_numpy()

##########################
#### Cluster Analysis ####
##########################

#### Elbow Plot To Determine Optimal Clusters ####
X=np.concatenate((k.reshape(len(k),1),mR.reshape(len(mR),1)),axis=1)
# Scale data
scaler=StandardScaler().fit(X)
XS=scaler.transform(X)

inertias=[]
distortions=[]
K_rng=range(1,15)
for i in K_rng:
    km=KMeans(n_clusters=i,max_iter=5000,random_state=seed).fit(XS)
    inertias.append(km.inertia_)
    distortions.append(sum(np.min(cdist(XS,km.cluster_centers_,'euclidean'),axis=1))
                        / XS.shape[0])
fig1=plt.figure(num=1)
ax1=plt.subplot(2,1,1)
ax1.plot(K_rng,inertias,'bx-')    
plt.xlabel('Number of Clusters')
plt.ylabel('Intertia')
ax2=plt.subplot(2,1,2)
ax2.plot(K_rng,distortions,'bx-')
plt.xlabel('Number of Clusters')
plt.ylabel('Distortion')

#### Optimal Cluster Number Based on Elbow ####
num_clust=3
km=KMeans(n_clusters=num_clust,max_iter=5000,random_state=seed).fit(XS)

#### MEANS
# Untransform cluster centers
cents=scaler.inverse_transform(km.cluster_centers_)
# Calculate center of k_e for clusters
temp=np.zeros((num_clust,1))
for i in range(num_clust):
    temp[i,0]=10**np.mean(np.log10(k_e[km.labels_==i]))
cents=np.hstack((cents,temp))  
# Assign population values
k_mean0=np.mean(k)
R_mean0=np.mean(mR)
k_e_mean0=10**np.mean(np.log10(k_e))

#### MEDIANS
med_cents=np.zeros((num_clust,3))
med_stds=np.zeros((num_clust,3))
for i in range(num_clust):
    med_cents[i,0]=np.median(k[km.labels_==i])
    med_cents[i,1]=np.median(mR[km.labels_==i])
    med_cents[i,2]=np.median(k_e[km.labels_==i])
    med_stds[i,0]=np.std(k[km.labels_==i])
    med_stds[i,1]=np.std(mR[km.labels_==i])
    med_stds[i,2]=np.std(k_e[km.labels_==i])
    
# Assign population values
k_median0=np.median(k)
R_median0=np.median(mR)
k_e_median0=np.median(k_e)
    

### Generate predictions for whole population STIM
cL0=stim.set_constants(R_mean0,k_e_mean0)
[Ks0,E0,E_err,Q_starc]=stim.stim_range(k_mean0,cL0,max_ksn=600)

cL1=stim.set_constants(R_median0,k_e_median0)
[Ks1,E1,E_err,Q_starc]=stim.stim_range(k_median0,cL1,max_ksn=2500)
KsanT=stim.an_thresh(E1,k_median0,cL1)
KsnoT=stim.an_const(E1,k_median0,cL1) 


### Calculate root mean squared error
def rmse(obs,pred):
    return np.sqrt(np.sum((obs-pred)**2)/len(obs))
def wrmse(obs,pred,weights):
    return np.sqrt(np.sum((1/weights**2)*((obs-pred)**2)))    

E_mean_pred=np.zeros(len(ksn))
E_med_pred=np.zeros(len(ksn))
for i in range(len(ksn)):
    [E_mean_pred[i],E_err,Q_starc]=stim.stim_one(ksn[i],k_mean0,cL0)
    [E_med_pred[i],E_err,Q_starc]=stim.stim_one(ksn[i],k_median0,cL1)
    
mean_rmse=rmse(e,E_mean_pred)
med_rmse=rmse(e,E_med_pred)
mean_rmse_w=wrmse(e,E_mean_pred,e_u)
med_rmse_w=wrmse(e,E_med_pred,e_u)    

### Switch For Inclusion
plot_mean=False
plot_median=True
plot_fit=False

### Calculate projected STIM
Ks=[]
E=[]
med_Ks=[]
med_E=[]
# phi=[]
cluster_rmse=[]
for i in range(num_clust):
    cL=stim.set_constants(cents[i,1],k_e_mean0)
    [Ks2,E2,E_err,Q_starc]=stim.stim_range(cents[i,0],cL,max_ksn=600)
    Ks.append(Ks2)
    E.append(E2)
    cL=stim.set_constants(med_cents[i,1],k_e_median0)
    [Ks3,E3,E_err,Q_starc]=stim.stim_range(med_cents[i,0],cL,max_ksn=600)
    med_Ks.append(Ks3)
    med_E.append(E3)
    
    clust_E_pred=np.zeros(len(ksn[km.labels_==i]))
    ksn_sel=ksn[km.labels_==i]
    for k in range(len(ksn[km.labels_==i])):
        [clust_E_pred[k],E_err,Q_starc]=stim.stim_one(ksn_sel[k],med_cents[i,0],cL)
    cluster_rmse.append(rmse(e[km.labels_==i],clust_E_pred))
        





if plot_fit:
    data_idx=e_u<e
    ### Real data without uncertainty
    Rdata=odr.RealData(ksn[data_idx],e[data_idx])
    ## Free R, k_e and k
    def stim_odr(B,x):
        cLi=stim.set_constants(B[0],B[1],omega_s=0.25)
        E=np.zeros(x.shape)
        for i in range(len(x)):
            [E[i],E_err,Q_starc]=stim.stim_one(x[i],B[2],cLi)
        return E
    stim_model=odr.Model(stim_odr)
    stim_odr=odr.ODR(Rdata,stim_model,beta0=[R_median0,k_e_median0,k_median0])
    stim_res=stim_odr.run()
    stim_R=stim_res.beta[0]
    stim_k_e=stim_res.beta[1]
    stim_k=stim_res.beta[2]
    
    stim_cL=stim.set_constants(stim_R,stim_k_e)
    [stim_Ks,stim_E,E_err,Q_starc]=stim.stim_range(stim_k,stim_cL,max_ksn=600)

   
### Start Plotting   
color_list=['maroon','dodgerblue','darkorange','crimson''darkolivegreen',]

# Fix k_e to population value
Ks_fix=[]
E_fix=[]
med_Ks_fix=[]
med_E_fix=[]
med_Ks_fix_noT=[]
med_Ks_fix_anT=[]
for i in range(num_clust):
    cL=stim.set_constants(cents[i,1],k_e_mean0)
    [Ks2,E2,E_err,Q_starc]=stim.stim_range(cents[i,0],cL,max_ksn=600)
    Ks_fix.append(Ks2)
    E_fix.append(E2)
    cL=stim.set_constants(med_cents[i,1],k_e_median0)
    [Ks3,E3,E_err,Q_starc]=stim.stim_range(med_cents[i,0],cL,max_ksn=2500)
    Ks4=stim.an_thresh(E3,med_cents[i,0],cL)
    Ks5=stim.an_const(E3,med_cents[i,0],cL)    
    med_Ks_fix.append(Ks3)
    med_E_fix.append(E3)
    med_Ks_fix_noT.append(Ks5)
    med_Ks_fix_anT.append(Ks4)

    
fig3=plt.figure(num=3,figsize=(12,12))
ax1=plt.subplot(2,2,1)
if plot_mean:
    ax1.plot(E0,Ks0,c='black',zorder=1,label='Whole Population - Mean',linewidth=2,linestyle=':')
if plot_median:
    ax1.plot(E1,Ks1,c='black',zorder=1,label='Whole Population - Median',linewidth=2)
if plot_fit:
    ax1.plot(stim_E,stim_Ks,c='black',zorder=1,linestyle=':',label='STIM Best Fit',linewidth=2)  
for i in range(num_clust):
    ax1.scatter(e[km.labels_==i],ksn[km.labels_==i],s=40,c=color_list[i],zorder=4)
    ax1.errorbar(e[km.labels_==i],ksn[km.labels_==i],yerr=ksn_u[km.labels_==i],
                 xerr=e_u[km.labels_==i],linestyle='none',c=color_list[i],zorder=3,elinewidth=0.5)    
    if plot_mean:
        ax1.plot(E_fix[i],Ks_fix[i],c=color_list[i],zorder=2,label='Mean Cluster '+str(i+1),linewidth=2,linestyle=':')
    if plot_median:
        ax1.plot(med_E_fix[i],med_Ks_fix[i],c=color_list[i],zorder=2,label='Median Cluster '+str(i+1),linewidth=2)


plt.xlim((0,8000))
plt.ylim((0,550))
plt.xlabel('Erosion Rate [m/Myr]')
plt.ylabel('$k_{sn}$ [m]')
plt.legend(loc='lower right')

ax1l=plt.subplot(2,2,2)
if plot_mean:
    ax1l.plot(E0,Ks0,c='black',zorder=1,label='Whole Population - Mean',linewidth=2,linestyle=':')
if plot_median:
    ax1l.plot(E1,Ks1,c='black',zorder=1,label='Whole Population - Median',linewidth=2)
if plot_fit:
    ax1l.plot(stim_E,stim_Ks,c='black',zorder=1,linestyle=':',label='STIM Best Fit',linewidth=2)     
for i in range(num_clust):
    ax1l.scatter(e[km.labels_==i],ksn[km.labels_==i],s=40,c=color_list[i],zorder=4)
    ax1l.errorbar(e[km.labels_==i],ksn[km.labels_==i],yerr=ksn_u[km.labels_==i],
                 xerr=e_u[km.labels_==i],linestyle='none',c=color_list[i],zorder=3,elinewidth=0.5)    
    if plot_mean:
        ax1l.plot(E_fix[i],Ks_fix[i],c=color_list[i],zorder=2,label='Mean Cluster '+str(i+1),linewidth=2,linestyle=':')
    if plot_median:
        ax1l.plot(med_E_fix[i],med_Ks_fix[i],c=color_list[i],zorder=2,label='Median Cluster '+str(i+1),linewidth=2)     
plt.xlim((10,10**4))
plt.xscale('log')
plt.ylim((0,550))
plt.xlabel('Erosion Rate [m/Myr]')
plt.ylabel('$k_{sn}$ [m]')

ax2=plt.subplot(2,2,3)
if plot_mean:
    ax2.scatter(R_mean0,k_mean0,c='black',marker='^',s=60,zorder=2)
if plot_median:
    ax2.scatter(R_median0,k_median0,c='black',marker='s',s=60,zorder=2)
    ax2.errorbar(R_median0,k_median0,xerr=np.std(mR),yerr=np.std(k),c='black',
                 zorder=2,linestyle='none')
if plot_fit:
    ax2.scatter(stim_R,stim_k,c='black',marker='*',s=60,zorder=2)       
for i in range(num_clust):
    if plot_mean:
        ax2.scatter(cents[i,1],cents[i,0],c=color_list[i],s=60,zorder=2,marker='^')
    if plot_median:
        ax2.scatter(med_cents[i,1],med_cents[i,0],c=color_list[i],s=60,zorder=2,marker='s')
        ax2.errorbar(med_cents[i,1],med_cents[i,0],xerr=med_stds[i,1],yerr=med_stds[i,0],c=color_list[i],
                     zorder=2,linestyle='none')        
    ax2.scatter(X[km.labels_==i,1],X[km.labels_==i,0],c=color_list[i],alpha=0.5,zorder=1)
plt.xlabel('Estimated Mean Runoff [mm/day]')
plt.ylabel('Estimated Variability')
plt.xlim((0,6))
plt.ylim((2,6))

ax3=plt.subplot(2,2,4)
if plot_mean:
    ax3.scatter(R_mean0,k_e_mean0,c='black',marker='^',s=60,zorder=2,label='Mean Center')
if plot_median:
    ax3.scatter(R_median0,k_e_median0,c='black',marker='s',s=60,zorder=2,
                label='Median; k={:.2f}'.format(k_median0)+
                '; R={:.2f}'.format(R_median0)+
                '; $k_e$={:.2e}'.format(k_e_median0))
if plot_fit:
    ax3.scatter(stim_R,stim_k_e,c='black',marker='*',s=60,zorder=2,
                label='Fit; k={:.2f}'.format(stim_k)+
                '; R={:.2f}'.format(stim_R)+
                '; $k_e$={:.2e}'.format(stim_k_e))
for i in range(num_clust):
    if plot_mean:
        ax3.scatter(cents[i,1],cents[i,2],c=color_list[i],s=60,zorder=2,marker='^')
    if plot_median:
        ax3.scatter(med_cents[i,1],med_cents[i,2],c=color_list[i],s=60,zorder=2,
                    marker='s',label='k={:.2f}'.format(med_cents[i,0])+
                    '; R={:.2f}'.format(med_cents[i,1])+
                    '; $k_e$={:.2e}'.format(med_cents[i,2]))     
    ax3.scatter(X[km.labels_==i,1],k_e[km.labels_==i],c=color_list[i],alpha=0.5,zorder=1)
plt.yscale('log')
plt.ylim((1e-13,1e-5))
# plt.ylim((1e-14,1e-9))
plt.xlabel('Estimated Mean Runoff [mm/day]')
plt.ylabel('Optimized $k_e$')
plt.legend(loc='upper right')
plt.xlim((0,6))
plt.suptitle('Fixed $k_e$')   
# plt.suptitle('Clustered $k_e$')



# plt.figure(num=10)
e_c_predicted=[]
e_c_high=[]
e_c_low=[]
for i in range(num_clust):
    Roi=med_cents[i,1]
    Rloi=med_cents[i,1]-med_stds[i,1]
    Rhoi=med_cents[i,1]+med_stds[i,1]
    koi=med_cents[i,0]
    kloi=med_cents[i,0]-med_stds[i,0]
    khoi=med_cents[i,0]+med_stds[i,0]
    
    cL=stim.set_constants(Roi,k_e_median0)
    cLl=stim.set_constants(Rloi,k_e_median0)
    cLh=stim.set_constants(Rhoi,k_e_median0)
    ksnOI=ksn[km.labels_==i]
    E_out=np.zeros(len(ksnOI))
    E_out_high=np.zeros(len(ksnOI)) # lower k and lower R
    E_out_low=np.zeros(len(ksnOI)) # higher k and higher R
    
    for j in range(len(ksnOI)):
        [E_out[j],E_err,Q_starc]=stim.stim_one(ksnOI[j],koi,cL)
        [E_out_high[j],E_err,Q_starc]=stim.stim_one(ksnOI[j],kloi,cLl)
        [E_out_low[j],E_err,Q_starc]=stim.stim_one(ksnOI[j],khoi,cLh)
    e_c_predicted.append(E_out)
    e_c_low.append(E_out_low)
    e_c_high.append(E_out_high)

e_pred=np.zeros(len(ksn))
e_alow=np.zeros(len(ksn))
e_ahigh=np.zeros(len(ksn))
cL=stim.set_constants(R_median0,k_e_median0)
cLl=stim.set_constants(R_median0-np.std(mR),k_e_median0)
cLh=stim.set_constants(R_median0+np.std(mR),k_e_median0)
for i in range(len(ksn)):
    [e_pred[i],E_err,Q_starc]=stim.stim_one(ksn[i],k_median0,cL)
    [e_ahigh[i],E_err,Q_starc]=stim.stim_one(ksn[i],k_median0-np.std(k),cLl)
    [e_alow[i],E_err,Q_starc]=stim.stim_one(ksn[i],k_median0+np.std(k),cLh)        
    
    

fig4=plt.figure(num=4,figsize=(15,8))
ax1=plt.subplot(1,2,1)
ax1.plot(np.logspace(0,5,25),np.logspace(0,5,25),linestyle=':',c='black',zorder=1)
for i in range(num_clust):
    ax1.scatter(e[km.labels_==i],e_c_predicted[i],s=40,c=color_list[i],zorder=3)
    ax1.errorbar(e[km.labels_==i],e_c_predicted[i],xerr=e_u[km.labels_==i],
                 linestyle='none',c=color_list[i],zorder=2,elinewidth=0.5)
    e_low=e_c_low[i]
    e_high=e_c_high[i]
    for j in range(len(e_low)):
        ax1.plot([e[km.labels_==i][j],e[km.labels_==i][j]],[e_low[j],e_high[j]],c=color_list[i],linewidth='0.5')
plt.xlabel('Observed Erosion Rate [m/Myr]')
plt.ylabel('Predicted Erosion Rate [m/Myr]')
plt.xscale('log')
plt.yscale('log')
plt.xlim((1,10**5))
plt.ylim((1,10**5))
plt.title('Clustered')

ax2=plt.subplot(1,2,2)
ax2.plot(np.logspace(0,5,25),np.logspace(0,5,25),linestyle=':',c='black',zorder=1)
for i in range(num_clust):
    ax2.scatter(e[km.labels_==i],e_pred[km.labels_==i],s=40,c=color_list[i],zorder=3)
    ax2.errorbar(e[km.labels_==i],e_pred[km.labels_==i],xerr=e_u[km.labels_==i],
                 linestyle='none',c=color_list[i],zorder=2,elinewidth=0.5)
    e_low=e_alow[km.labels_==i]
    e_high=e_ahigh[km.labels_==i]
    for j in range(len(e_low)):
        ax2.plot([e[km.labels_==i][j],e[km.labels_==i][j]],[e_low[j],e_high[j]],c=color_list[i],linewidth='0.5')
plt.xlabel('Observed Erosion Rate [m/Myr]')
plt.ylabel('Predicted Erosion Rate [m/Myr]')
plt.xscale('log')
plt.yscale('log') 
plt.xlim((1,10**5))
plt.ylim((1,10**5))
plt.title('Median')


    
fig5=plt.figure(num=5,figsize=(12,16))

ax1=plt.subplot(3,2,1)
ax1.plot(E1,Ks1,c='black',zorder=1,label='Numerical Integration',linewidth=2)
ax1.plot(E1,KsanT,c='black',zorder=1,label='Analytical w/Threshold',linewidth=2,linestyle=':')
ax1.plot(E1,KsnoT,c='black',zorder=1,label='Constant Discharge',linewidth=2,linestyle='--')  
ax1.scatter(e,ksn,s=40,c='black',zorder=3)
ax1.errorbar(e,ksn,yerr=ksn_u,xerr=e_u,linestyle='none',c='black',zorder=2,elinewidth=0.5)
plt.xlabel('Erosion Rate [m/Myr]')
plt.xlim((0,8000))
plt.ylabel('$k_{sn}$ [m]')
plt.ylim((0,1350))
plt.title('All Data')
plt.legend(loc='lower right')

ax2=plt.subplot(3,2,2)
i=0
ax2.plot(med_E_fix[i],med_Ks_fix[i],c=color_list[i],zorder=1,label='Numerical Integration',linewidth=2)
ax2.plot(med_E_fix[i],med_Ks_fix_anT[i],c=color_list[i],zorder=1,label='Analytical w/Threshold',linewidth=2,linestyle=':')
ax2.plot(med_E_fix[i],med_Ks_fix_noT[i],c=color_list[i],zorder=1,label='Constant Discharge',linewidth=2,linestyle='--')  
ax2.scatter(e,ksn,s=40,c='gray',zorder=3)
ax2.errorbar(e,ksn,yerr=ksn_u,xerr=e_u,linestyle='none',c='gray',zorder=2,elinewidth=0.5)
ax2.scatter(e[km.labels_==i],ksn[km.labels_==i],s=40,c=color_list[i],zorder=5)
ax2.errorbar(e[km.labels_==i],ksn[km.labels_==i],yerr=ksn_u[km.labels_==i],
              xerr=e_u[km.labels_==i],linestyle='none',c=color_list[i],zorder=4,elinewidth=0.5)
plt.xlabel('Erosion Rate [m/Myr]')
plt.xlim((0,8000))
plt.ylabel('$k_{sn}$ [m]')
plt.ylim((0,1350))
plt.legend(loc='lower right')
plt.title('Cluster '+str(i+1))

ax2=plt.subplot(3,2,3)
i=1
ax2.plot(med_E_fix[i],med_Ks_fix[i],c=color_list[i],zorder=1,label='Numerical Integration',linewidth=2)
ax2.plot(med_E_fix[i],med_Ks_fix_anT[i],c=color_list[i],zorder=1,label='Analytical w/Threshold',linewidth=2,linestyle=':')
ax2.plot(med_E_fix[i],med_Ks_fix_noT[i],c=color_list[i],zorder=1,label='Constant Discharge',linewidth=2,linestyle='--')  
ax2.scatter(e,ksn,s=40,c='gray',zorder=3)
ax2.errorbar(e,ksn,yerr=ksn_u,xerr=e_u,linestyle='none',c='gray',zorder=2,elinewidth=0.5)
ax2.scatter(e[km.labels_==i],ksn[km.labels_==i],s=40,c=color_list[i],zorder=5)
ax2.errorbar(e[km.labels_==i],ksn[km.labels_==i],yerr=ksn_u[km.labels_==i],
              xerr=e_u[km.labels_==i],linestyle='none',c=color_list[i],zorder=4,elinewidth=0.5)
plt.xlabel('Erosion Rate [m/Myr]')
plt.xlim((0,8000))
plt.ylabel('$k_{sn}$ [m]')
plt.ylim((0,1350))
plt.legend(loc='lower right')
plt.title('Cluster '+str(i+1))

ax2=plt.subplot(3,2,4)
i=2
ax2.plot(med_E_fix[i],med_Ks_fix[i],c=color_list[i],zorder=1,label='Numerical Integration',linewidth=2)
ax2.plot(med_E_fix[i],med_Ks_fix_anT[i],c=color_list[i],zorder=1,label='Analytical w/Threshold',linewidth=2,linestyle=':')
ax2.plot(med_E_fix[i],med_Ks_fix_noT[i],c=color_list[i],zorder=1,label='Constant Discharge',linewidth=2,linestyle='--')  
ax2.scatter(e,ksn,s=40,c='gray',zorder=3)
ax2.errorbar(e,ksn,yerr=ksn_u,xerr=e_u,linestyle='none',c='gray',zorder=2,elinewidth=0.5)
ax2.scatter(e[km.labels_==i],ksn[km.labels_==i],s=40,c=color_list[i],zorder=5)
ax2.errorbar(e[km.labels_==i],ksn[km.labels_==i],yerr=ksn_u[km.labels_==i],
              xerr=e_u[km.labels_==i],linestyle='none',c=color_list[i],zorder=4,elinewidth=0.5)
plt.xlabel('Erosion Rate [m/Myr]')
plt.xlim((0,8000))
plt.ylabel('$k_{sn}$ [m]')
plt.ylim((0,1350))
plt.legend(loc='lower right')
plt.title('Cluster '+str(i+1))

ax3=plt.subplot(3,1,3)
ax3.plot(E1,Ks1,c='black',zorder=1,label='Numerical Integration',linewidth=2)
ax3.plot(E1,KsanT,c='black',zorder=1,label='Analytical w/Threshold',linewidth=2,linestyle=':')
ax3.plot(E1,KsnoT,c='black',zorder=1,label='Constant Discharge',linewidth=2,linestyle='--')  
for i in range(num_clust):
    ax3.plot(med_E_fix[i],med_Ks_fix[i],c=color_list[i],zorder=1,label='Numerical Integration',linewidth=2)
    ax3.plot(med_E_fix[i],med_Ks_fix_anT[i],c=color_list[i],zorder=1,label='Analytical w/Threshold',linewidth=2,linestyle=':')
    ax3.plot(med_E_fix[i],med_Ks_fix_noT[i],c=color_list[i],zorder=1,label='Constant Discharge',linewidth=2,linestyle='--') 
plt.xlabel('Erosion Rate [m/Myr]')
plt.ylabel('$k_{sn}$ [m]')
plt.xlim((0,1.5e6))


##PLOT TECTONICS
# Import convergence
cdf=pd.read_csv('data_tables/interpolated_convergence.txt')
c=cdf['con_vel'].to_numpy()
chi=cdf['con_vel_hi'].to_numpy()-c
clo=c-cdf['con_vel_lo'].to_numpy()
l=cdf['lc_vel'].to_numpy()
lhi=cdf['lc_vel_hi'].to_numpy()-l
llo=l-cdf['lc_vel_lo'].to_numpy()
g=cdf['gc_vel'].to_numpy()
ghi=cdf['gc_vel_hi'].to_numpy()-g
glo=g-cdf['gc_vel_lo'].to_numpy()
# Import swath distances
sdf=pd.read_csv('data_tables/swath_distances.txt')
ds=sdf['D_along_SW'].to_numpy()
db=sdf['D_from_SW'].to_numpy()

# Structure error
# Build Indices
idx=np.where((df['sample_name']!='16AF01') & (df['region']!='LC'))
idx_ngc=np.where((df['sample_name']=='16AF01'))
idx_lc=np.where((df['region']=='LC'))
# Structure error
c_u=np.vstack((clo[idx].reshape(1,len(clo[idx])),chi[idx].reshape(1,len(chi[idx]))))
c_u_ngc=np.vstack((clo[idx_ngc].reshape(1,len(clo[idx_ngc])),chi[idx_ngc].reshape(1,len(chi[idx_ngc]))))
c_u_lc=np.vstack((clo[idx_lc].reshape(1,len(clo[idx_lc])),chi[idx_lc].reshape(1,len(chi[idx_lc]))))
# Index remaining
c_m=c[idx]
e_m=e[idx]
e_u_m=e_u[idx]
db_m=db[idx]
l_m=km.labels_[idx]
c_ngc=c[idx_ngc]
e_ngc=e[idx_ngc]
e_u_ngc=e_u[idx_ngc]
db_ngc=db[idx_ngc]
l_ngc=km.labels_[idx_ngc]
c_lc=c[idx_lc]
e_lc=e[idx_lc]
e_u_lc=e_u[idx_lc]
db_lc=db[idx_lc]
l_lc=km.labels_[idx_lc]

from matplotlib import gridspec

fig6=plt.figure(num=6,figsize=(15,5))
gs=gridspec.GridSpec(1,2,width_ratios=[1,2])
ax1=plt.subplot(gs[0])
for i in range(num_clust):
    ax1.scatter(c_m[l_m==i],e_m[l_m==i],s=60,c=color_list[i],zorder=3,edgecolors='k')
    ax1.errorbar(c_m[l_m==i],e_m[l_m==i],xerr=c_u[:,l_m==i],yerr=e_u_m[l_m==i],linestyle='none',c='black',zorder=1,elinewidth=0.5)
    
    if np.any(l_ngc==i): 
        ax1.scatter(c_ngc[l_ngc==i],e_ngc[l_ngc==i],s=60,c=color_list[i],zorder=3,edgecolors='k',marker='s')
        ax1.errorbar(c_ngc[l_ngc==i],e_ngc[l_ngc==i],xerr=c_u_ngc[:,l_ngc==i],yerr=e_u_ngc[l_ngc==i],linestyle='none',c='black',zorder=1,elinewidth=0.5)
    
    if np.any(l_lc==i):
        ax1.scatter(c_lc[l_lc==i],e_lc[l_lc==i],s=60,c=color_list[i],zorder=3,edgecolors='k',marker='^')
        ax1.errorbar(c_lc[l_lc==i],e_lc[l_lc==i],xerr=c_u_lc[:,l_lc==i],yerr=e_u_lc[l_lc==i],linestyle='none',c='black',zorder=1,elinewidth=0.5)

ax1.set_xlabel('GC-LC Convergence [mm/yr]')
ax1.set_ylabel('Erosion Rate [m/Myr]')
ax1.set_yscale('log')
ax1.set_ylim((10,10**4))
ax1.set_xlim((0,12))


ax2=plt.subplot(gs[1])
for i in range(num_clust):
    ax2.scatter(-db_m[l_m==i],e_m[l_m==i],s=60,c=color_list[i],zorder=2,edgecolors='k')
    ax2.errorbar(-db_m[l_m==i],e_m[l_m==i],yerr=e_u_m[l_m==i],linestyle='none',c='black',zorder=1,elinewidth=0.5)
    
    if np.any(l_ngc==i):
        ax2.scatter(db_ngc[l_ngc==i],e_ngc[l_ngc==i],s=60,c=color_list[i],zorder=2,edgecolors='k',marker='s')
        ax2.errorbar(db_ngc[l_ngc==i],e_ngc[l_ngc==i],yerr=e_u_ngc[l_ngc==i],linestyle='none',c='black',zorder=1,elinewidth=0.5)        

    if np.any(l_lc==i):
        ax2.scatter(-db_lc[l_lc==i],e_lc[l_lc==i],s=60,c=color_list[i],zorder=2,edgecolors='k',marker='^')
        ax2.errorbar(-db_lc[l_lc==i],e_lc[l_lc==i],yerr=e_u_lc[l_lc==i],linestyle='none',c='black',zorder=1,elinewidth=0.5)

ax2.set_xlabel('Distance from Center [km]')
ax2.set_ylabel('Erosion Rate [m/Myr]')
ax2.set_yscale('log')
ax2.set_ylim((10,10**4))

## PLOT CONCAVITY
theta=df['trunk_concavity'].to_numpy()
ksn_std=df['std_ksn'].to_numpy()

fig7=plt.figure(num=7,figsize=(15,8))
ax1=plt.subplot(1,2,1)
for i in range(num_clust):
    ax1.scatter(ksn[km.labels_==i],theta[km.labels_==i],c=color_list[i],s=60,
                edgecolors='k',zorder=2)
ax1.errorbar(ksn,theta,xerr=ksn_u,linestyle='none',c='black',zorder=1,elinewidth=0.5)
plt.xlabel('$k_{sn}$ [m]')
plt.ylabel('Best Fit Trunk Concavity')

ax2=plt.subplot(1,2,2)
for i in range(num_clust):
    ax2.scatter(e[km.labels_==i],theta[km.labels_==i],c=color_list[i],s=60,
                edgecolors='k',zorder=2)
ax2.errorbar(e,theta,xerr=e_u,linestyle='none',c='black',zorder=1,elinewidth=0.5)
plt.xscale('log')
plt.xlim((10,10**4))
plt.xlabel('Erosion Rate [m/Myr]')
plt.ylabel('Best Fit Trunk Concavity')

