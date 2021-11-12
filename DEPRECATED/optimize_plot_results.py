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

cdf=pd.read_csv('data_tables/interpolated_convergence.txt')
v=cdf['con_vel'].to_numpy()

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


##################################
#### Plot Optimization of k_e ####
##################################
# Load in k_e optimization results
dfke=pd.read_csv('result_tables/k_e_optim.csv')
k_e=dfke['k_e_boostrap'].to_numpy()
k_e_simple=dfke['k_e_simple'].to_numpy()
est_err=dfke['rmse_bootstrap'].to_numpy()
e_predicted=dfke['E_pred'].to_numpy() 


fig1=plt.figure(num=1,figsize=(12,10))
kenorm=colors.Normalize(vmin=-12,vmax=-6)
ax1=plt.subplot(2,2,1)
plt.plot(np.array([0,8000]),np.array([0,8000]),c='black',linestyle=':',zorder=1)
plt.errorbar(e,e_predicted,xerr=e_u,yerr=est_err,linestyle='none',c='black',
              zorder=2,elinewidth=0.5)
sc=plt.scatter(e,e_predicted,s=40,c=np.log10(k_e),zorder=3,norm=kenorm,cmap='magma')
cbar1=plt.colorbar(sc)
cbar1.ax.set_ylabel('Log Optimized $k_e$')
plt.xlabel('Observed Erosion Rate [m/Myr]')
plt.ylabel('Predicted Erosion Rate - Bootstrap Optimized $k_e$ [m/Myr]')
plt.title('Results for '+k_var)

ax2=plt.subplot(2,2,2)
sc2=ax2.scatter(mR,k,s=40,c=np.log10(k_e),norm=kenorm,zorder=1,cmap='magma')
cbar2=plt.colorbar(sc2,ax=ax2)
cbar2.ax.set_ylabel('Log Optimized $k_e$')
plt.xlabel('Estimated Mean Runoff [mm/day]')
plt.ylabel('Estimated Variability')

knorm=colors.Normalize(vmin=2,vmax=6)
ax3=plt.subplot(2,2,3)
sc3=ax3.scatter(mR,k_e,s=40,c=k,norm=knorm,zorder=2)
cbar3=plt.colorbar(sc3,ax=ax3)
cbar3.ax.set_ylabel('Estimated Variability (k)')
plt.yscale('log')
plt.ylim((1e-13,1e-5))
plt.xlabel('Estimated Mean Runoff [mm/day]')
plt.ylabel('Optimized $k_e$')

ax3_2=ax3.twinx()
ax3_2.set_ylabel(r'Optimized $\Psi_c$')
ax3_2.set_yscale('log')
low_psi=(1e-13)*(45**1.5)
hi_psi=(1e-5)*(45**1.5)
ax3_2.set_ylim((low_psi,hi_psi))
ax3_2.set_ylabel(r'Optimized $\Psi_c$')


## Plots comparing k_e and lithology
plnorm=colors.Normalize(vmin=0,vmax=100)
fig2=plt.figure(num=2,figsize=(15,15))
# Percent Carb
ax1a=plt.subplot(4,2,1)
ax1a.scatter(e,ksn,s=40,c=perc_carb,zorder=3,norm=plnorm,cmap='plasma')
ax1a.errorbar(e,ksn,yerr=ksn_u,xerr=e_u,
              linestyle='none',c='black',zorder=2,elinewidth=0.5)
plt.xlim((10,10**4))
plt.ylim((0,600))
plt.xscale('log')
plt.xlabel('Erosion Rate [m/Myr]')
plt.ylabel('$k_{sn}$ [m]')
ax1b=plt.subplot(4,2,2)
sc1=ax1b.scatter(mR,k_e,s=40,c=perc_carb,norm=plnorm,cmap='plasma')
plt.yscale('log')
plt.ylim((1e-13,1e-5))
cbar1=plt.colorbar(sc1,ax=ax1b)
cbar1.ax.set_ylabel('Dominant Carbonate [%]')
plt.xlabel('Estimated Mean Runoff [mm/day]')
plt.ylabel('Optimized $k_e$')
# Percent Clastic
ax2a=plt.subplot(4,2,3)
ax2a.scatter(e,ksn,s=40,c=perc_clastic,zorder=3,norm=plnorm,cmap='plasma')
ax2a.errorbar(e,ksn,yerr=ksn_u,xerr=e_u,
              linestyle='none',c='black',zorder=2,elinewidth=0.5)
plt.xlim((10,10**4))
plt.ylim((0,600))
plt.xscale('log')
plt.xlabel('Erosion Rate [m/Myr]')
plt.ylabel('$k_{sn}$ [m]')
ax2=plt.subplot(4,2,4)
sc2=ax2.scatter(mR,k_e,s=40,c=perc_clastic,norm=plnorm,cmap='plasma')
plt.yscale('log')
plt.ylim((1e-13,1e-5))
cbar2=plt.colorbar(sc2,ax=ax2)
cbar2.ax.set_ylabel('Dominant Clastic [%]')
plt.xlabel('Estimated Mean Runoff [mm/day]')
plt.ylabel('Optimized $k_e$')
# Percent Sandstone and Shale
ax3a=plt.subplot(4,2,5)
ax3a.scatter(e,ksn,s=40,c=perc_ss,zorder=3,norm=plnorm,cmap='plasma')
ax3a.errorbar(e,ksn,yerr=ksn_u,xerr=e_u,
              linestyle='none',c='black',zorder=2,elinewidth=0.5)
plt.xlim((10,10**4))
plt.ylim((0,600))
plt.xscale('log')
plt.xlabel('Erosion Rate [m/Myr]')
plt.ylabel('$k_{sn}$ [m]')
ax3=plt.subplot(4,2,6)
sc3=ax3.scatter(mR,k_e,s=40,c=perc_ss,norm=plnorm,cmap='plasma')
plt.yscale('log')
plt.ylim((1e-13,1e-5))
cbar3=plt.colorbar(sc3,ax=ax3)
cbar3.ax.set_ylabel('Sandstone & Shale [%]')
plt.xlabel('Estimated Mean Runoff [mm/day]')
plt.ylabel('Optimized $k_e$')
# Percent IgMet
ax4a=plt.subplot(4,2,7)
ax4a.scatter(e,ksn,s=40,c=perc_igmet,zorder=3,norm=plnorm,cmap='plasma')
ax4a.errorbar(e,ksn,yerr=ksn_u,xerr=e_u,
              linestyle='none',c='black',zorder=2,elinewidth=0.5)
plt.xlim((10,10**4))
plt.ylim((0,600))
plt.xscale('log')
plt.xlabel('Erosion Rate [m/Myr]')
plt.ylabel('$k_{sn}$ [m]')
ax4=plt.subplot(4,2,8)
sc4=ax4.scatter(mR,k_e,s=40,c=perc_igmet,norm=plnorm,cmap='plasma')
plt.yscale('log')
plt.ylim((1e-13,1e-5))
cbar4=plt.colorbar(sc4,ax=ax4)
cbar4.ax.set_ylabel('Igneous - Metamorphic [%]')
plt.xlabel('Estimated Mean Runoff [mm/day]')
plt.ylabel('Optimized $k_e$')


####################################
#### Plot Optimization of tau_c ####
####################################

# Load in tau_c optimization results
dftc=pd.read_csv('result_tables/tau_c_optim.csv')
tau_c=dftc['tau_c_boostrap'].to_numpy()
tau_c_simple=dftc['tau_c_simple'].to_numpy()
est_err=dfke['rmse_bootstrap'].to_numpy()
e_predicted=dfke['E_pred'].to_numpy()

k_e_fix=2.24e-10


fig3=plt.figure(num=3,figsize=(12,10))
tcnorm=colors.Normalize(vmin=10,vmax=90)
ax1=plt.subplot(2,2,1)
plt.plot(np.array([0,8000]),np.array([0,8000]),c='black',linestyle=':',zorder=1)
plt.errorbar(e,e_predicted,xerr=e_u,yerr=est_err,linestyle='none',c='black',
              zorder=2,elinewidth=0.5)
sc=plt.scatter(e,e_predicted,s=40,c=tau_c,zorder=3,norm=tcnorm,cmap='inferno')
cbar1=plt.colorbar(sc)
cbar1.ax.set_ylabel(r'Optimized $\tau_c$')
plt.xlabel('Observed Erosion Rate [m/Myr]')
plt.ylabel('Predicted Erosion Rate - Bootstrap Optimized $k_e$ [m/Myr]')
plt.title('Results for '+k_var)

ax2=plt.subplot(2,2,2)
sc2=ax2.scatter(mR,k,s=40,c=tau_c,norm=tcnorm,zorder=1,cmap='inferno')
cbar2=plt.colorbar(sc2,ax=ax2)
cbar2.ax.set_ylabel(r'Optimized $\tau_c$')
plt.xlabel('Estimated Mean Runoff [mm/day]')
plt.ylabel('Estimated Variability')


knorm=colors.Normalize(vmin=2,vmax=6)

ax3=plt.subplot(2,2,3)
sc3=ax3.scatter(mR,tau_c,s=40,c=k,norm=knorm,zorder=2)
cbar3=plt.colorbar(sc3,ax=ax3)
cbar3.ax.set_ylabel('Estimated Variability (k)')
plt.yscale('log')
plt.ylim((10,100))
plt.xlabel('Estimated Mean Runoff [mm/day]')
plt.ylabel(r'Optimized $\tau_c$')

ax3_2=ax3.twinx()
ax3_2.set_ylabel(r'Optimized $\Psi_c$')
ax3_2.set_yscale('log')
low_psi=(k_e_fix)*(10**1.5)
hi_psi=(k_e_fix)*(100**1.5)
ax3_2.set_ylim((low_psi,hi_psi))
ax3_2.set_ylabel(r'Optimized $\Psi_c$')


fig4=plt.figure(num=4,figsize=(12,10))
kenorm=colors.Normalize(vmin=-12,vmax=-6)
ax1=plt.subplot(2,2,1)
sc2=ax1.scatter(mR,k,s=40,c=np.log10(k_e),norm=kenorm,zorder=1,cmap='magma')
cbar2=plt.colorbar(sc2,ax=ax1)
cbar2.ax.set_ylabel('Log Optimized $k_e$')
plt.xlabel('Estimated Mean Runoff [mm/day]')
plt.ylabel('Estimated Variability')

knorm=colors.Normalize(vmin=2,vmax=6)
ax2=plt.subplot(2,2,2)
sc3=ax2.scatter(mR,k_e,s=40,c=k,norm=knorm,zorder=2)
ax2.axhline(k_e_fix,c='black',linestyle=':')
cbar3=plt.colorbar(sc3,ax=ax2)
cbar3.ax.set_ylabel('Estimated Variability (k)')
plt.yscale('log')
plt.ylim((1e-13,1e-5))
plt.xlabel('Estimated Mean Runoff [mm/day]')
plt.ylabel('Optimized $k_e$')

ax2_2=ax2.twinx()
ax2_2.set_ylabel(r'Optimized $\Psi_c$')
ax2_2.set_yscale('log')
low_psi=(1e-13)*(45**1.5)
hi_psi=(1e-5)*(45**1.5)
ax2_2.set_ylim((low_psi,hi_psi))
ax2_2.set_ylabel(r'Optimized $\Psi_c$')

ax3=plt.subplot(2,2,3)
tcnorm=colors.Normalize(vmin=10,vmax=90)
sc2=ax3.scatter(mR,k,s=40,c=tau_c,norm=tcnorm,zorder=1,cmap='inferno')
cbar2=plt.colorbar(sc2,ax=ax3)
cbar2.ax.set_ylabel(r'Optimized $\tau_c$')
plt.xlabel('Estimated Mean Runoff [mm/day]')
plt.ylabel('Estimated Variability')

knorm=colors.Normalize(vmin=2,vmax=6)
ax4=plt.subplot(2,2,4)
sc3=ax4.scatter(mR,tau_c,s=40,c=k,norm=knorm,zorder=2)
ax4.axhline(45,c='black',linestyle=':')
cbar3=plt.colorbar(sc3,ax=ax4)
cbar3.ax.set_ylabel('Estimated Variability (k)')
plt.yscale('log')
plt.ylim((10,100))
plt.xlabel('Estimated Mean Runoff [mm/day]')
plt.ylabel(r'Optimized $\tau_c$')

ax4_2=ax4.twinx()
ax4_2.set_ylabel(r'Optimized $\Psi_c$')
ax4_2.set_yscale('log')
low_psi=(k_e_fix)*(10**1.5)
hi_psi=(k_e_fix)*(100**1.5)
ax4_2.set_ylim((low_psi,hi_psi))
ax4_2.set_ylabel(r'Optimized $\Psi_c$')




## Plots comparing k_e and lithology
plnorm=colors.Normalize(vmin=0,vmax=100)
fig5=plt.figure(num=5,figsize=(15,15))
# Percent Carb
ax1a=plt.subplot(4,2,1)
ax1a.scatter(e,ksn,s=40,c=perc_carb,zorder=3,norm=plnorm)
ax1a.errorbar(e,ksn,yerr=ksn_u,xerr=e_u,
              linestyle='none',c='black',zorder=2,elinewidth=0.5)
plt.xlim((10,10**4))
plt.ylim((0,600))
plt.xscale('log')
plt.xlabel('Erosion Rate [m/Myr]')
plt.ylabel('$k_{sn}$ [m]')
ax1b=plt.subplot(4,2,2)
sc1=ax1b.scatter(mR,tau_c,s=40,c=perc_carb,norm=plnorm)
plt.ylim((10,90))
cbar1=plt.colorbar(sc1,ax=ax1b)
cbar1.ax.set_ylabel('Dominant Carbonate [%]')
plt.xlabel('Estimated Mean Runoff [mm/day]')
plt.ylabel(r'Optimized $\tau_c$')
# Percent Clastic
ax2a=plt.subplot(4,2,3)
ax2a.scatter(e,ksn,s=40,c=perc_clastic,zorder=3,norm=plnorm)
ax2a.errorbar(e,ksn,yerr=ksn_u,xerr=e_u,
              linestyle='none',c='black',zorder=2,elinewidth=0.5)
plt.xlim((10,10**4))
plt.ylim((0,600))
plt.xscale('log')
plt.xlabel('Erosion Rate [m/Myr]')
plt.ylabel('$k_{sn}$ [m]')
ax2=plt.subplot(4,2,4)
sc2=ax2.scatter(mR,tau_c,s=40,c=perc_clastic,norm=plnorm)
plt.ylim((10,90))
cbar2=plt.colorbar(sc2,ax=ax2)
cbar2.ax.set_ylabel('Dominant Clastic [%]')
plt.xlabel('Estimated Mean Runoff [mm/day]')
plt.ylabel(r'Optimized $\tau_c$')
# Percent Sandstone and Shale
ax3a=plt.subplot(4,2,5)
ax3a.scatter(e,ksn,s=40,c=perc_ss,zorder=3,norm=plnorm)
ax3a.errorbar(e,ksn,yerr=ksn_u,xerr=e_u,
              linestyle='none',c='black',zorder=2,elinewidth=0.5)
plt.xlim((10,10**4))
plt.ylim((0,600))
plt.xscale('log')
plt.xlabel('Erosion Rate [m/Myr]')
plt.ylabel('$k_{sn}$ [m]')
ax3=plt.subplot(4,2,6)
sc3=ax3.scatter(mR,tau_c,s=40,c=perc_ss,norm=plnorm)
plt.ylim((10,90))
cbar3=plt.colorbar(sc3,ax=ax3)
cbar3.ax.set_ylabel('Sandstone & Shale [%]')
plt.xlabel('Estimated Mean Runoff [mm/day]')
plt.ylabel(r'Optimized $\tau_c$')
# Percent IgMet
ax4a=plt.subplot(4,2,7)
ax4a.scatter(e,ksn,s=40,c=perc_igmet,zorder=3,norm=plnorm)
ax4a.errorbar(e,ksn,yerr=ksn_u,xerr=e_u,
              linestyle='none',c='black',zorder=2,elinewidth=0.5)
plt.xlim((10,10**4))
plt.ylim((0,600))
plt.xscale('log')
plt.xlabel('Erosion Rate [m/Myr]')
plt.ylabel('$k_{sn}$ [m]')
ax4=plt.subplot(4,2,8)
sc4=ax4.scatter(mR,tau_c,s=40,c=perc_igmet,norm=plnorm)
plt.ylim((10,90))
cbar4=plt.colorbar(sc4,ax=ax4)
cbar4.ax.set_ylabel('Igneous - Metamorphic [%]')
plt.xlabel('Estimated Mean Runoff [mm/day]')
plt.ylabel(r'Optimized $\tau_c$')

