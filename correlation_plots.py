#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Broad check as to whether topography (as either ksn or gradient) or erosion
rate is correlated with a wide array of climatic or lithologic factors.

Written by Adam M. Forte for 
"Low variability runoff inhibits coupling of climate, tectonics, and 
topography in the Greater Caucasus"

If you use this code or derivatives, please cite the original paper.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors
from scipy.stats import pearsonr

# Read in data
df=pd.read_csv('data_tables/gc_ero_master_table.csv')
# Extract main variables of interest
ksn=df['mean_ksn'].to_numpy()
ksn_u=df['se_ksn'].to_numpy()
e=df['St_E_rate_m_Myr'].to_numpy()
e_u=df['St_Ext_Unc'].to_numpy()
mG=df['mean_gradient'].to_numpy()
mG_u=df['se_gradient'].to_numpy()


mRa=df['corrected_mean_trmm'].to_numpy()
mS=df['mean_SNOWstd'].to_numpy()
da=df['drainage_area'].to_numpy()
mnZ=df['mean_el'].to_numpy()
mxZ=df['max_el'].to_numpy()
tC=df['trunk_concavity'].to_numpy()


# Assemble stats (these do not sum to 100, mixed and matched)
perc_carb=(df['rock_type_perc_limestone_minor_clastics'].to_numpy()+
         df['rock_type_perc_limestone_marl'].to_numpy()+
         df['rock_type_perc_limestone'].to_numpy()+
         df['rock_type_perc_marl'].to_numpy())
perc_clastic=(df['rock_type_perc_sandstone_shale'].to_numpy()+
              df['rock_type_perc_sandstone_minor_carbonate'].to_numpy()+
              df['rock_type_perc_shale'].to_numpy()+
              df['rock_type_perc_conglomerates'].to_numpy())
perc_igmet=(df['rock_type_perc_phyllite_schist'].to_numpy()+
            df['rock_type_perc_volcanic_undiff'].to_numpy()+
            df['rock_type_perc_granite'].to_numpy())

# Set colorscales and maps
e_norm=colors.Normalize(vmin=1,vmax=4)
ksn_norm=colors.Normalize(vmin=0,vmax=500)
g_norm=colors.Normalize(vmin=0,vmax=0.7)
e_map='magma'
ksn_map='cividis'

# Initiate Figure
fig1=plt.figure(num=1,figsize=(20,40))
################
# Drainage Area
[r,p]=pearsonr(ksn,np.log10(da))
lab_str='r = '+str(np.round(r,3))+'; p = '+str(np.round(p,3))
ax11=plt.subplot(9,3,1)
sc1=ax11.scatter(ksn,np.log10(da),s=40,c=np.log10(e),norm=e_norm,cmap=e_map,zorder=2,
                 label=lab_str)
ax11.errorbar(ksn,np.log10(da),xerr=ksn_u,linestyle='none',c='black',zorder=1,
              elinewidth=0.5)
cbar1=plt.colorbar(sc1,ax=ax11)
cbar1.ax.set_ylabel('Log Erosion Rate [m/Myr]')
plt.xlim((0,550))
# plt.xlabel('Mean $k_{sn}$ [m]')
plt.ylabel('Log Drainage Area [$km^2$]')
plt.ylim((0.5,2.5))
plt.legend(loc='best')

[r,p]=pearsonr(mG,np.log10(da))
lab_str='r = '+str(np.round(r,3))+'; p = '+str(np.round(p,3))
ax12=plt.subplot(9,3,2)
sc1=ax12.scatter(mG,np.log10(da),s=40,c=np.log10(e),norm=e_norm,cmap=e_map,zorder=2,
                 label=lab_str)
ax12.errorbar(mG,np.log10(da),xerr=mG_u,linestyle='none',c='black',zorder=1,
              elinewidth=0.5)
cbar1=plt.colorbar(sc1,ax=ax12)
cbar1.ax.set_ylabel('Log Erosion Rate [m/Myr]')
plt.xlim((0,0.8))
# plt.xlabel('Mean Gradient [m/m]')
plt.ylim((0.5,2.5))
plt.legend(loc='best')

[r,p]=pearsonr(e,np.log10(da))
lab_str='r = '+str(np.round(r,3))+'; p = '+str(np.round(p,3))
ax13=plt.subplot(9,3,3)
sc1=ax13.scatter(e,np.log10(da),s=40,c=ksn,norm=ksn_norm,cmap=ksn_map,zorder=2,
                 label=lab_str)
ax13.errorbar(e,np.log10(da),xerr=e_u,linestyle='none',c='black',zorder=1,
              elinewidth=0.5)
cbar1=plt.colorbar(sc1,ax=ax13)
cbar1.ax.set_ylabel('Mean $k_{sn}$ [m]')
plt.xlim((10,10000))
# plt.xlabel('Erosion Rate [m/Myr]')
plt.xscale('log')
plt.ylim((0.5,2.5))
plt.legend(loc='best')

################
# Trunk Concavity
[r,p]=pearsonr(ksn,tC)
lab_str='r = '+str(np.round(r,3))+'; p = '+str(np.round(p,3))
ax21=plt.subplot(9,3,4)
sc1=ax21.scatter(ksn,tC,s=40,c=np.log10(e),norm=e_norm,cmap=e_map,zorder=2,
                 label=lab_str)
ax21.errorbar(ksn,tC,xerr=ksn_u,linestyle='none',c='black',zorder=1,
              elinewidth=0.5)
cbar1=plt.colorbar(sc1,ax=ax21)
cbar1.ax.set_ylabel('Log Erosion Rate [m/Myr]')
plt.xlim((0,550))
# plt.xlabel('Mean $k_{sn}$ [m]')
plt.ylabel('Trunk Concavity')
plt.ylim((0,0.8))
plt.legend(loc='best')

[r,p]=pearsonr(mG,tC)
lab_str='r = '+str(np.round(r,3))+'; p = '+str(np.round(p,3))
ax22=plt.subplot(9,3,5)
sc1=ax22.scatter(mG,tC,s=40,c=np.log10(e),norm=e_norm,cmap=e_map,zorder=2,
                 label=lab_str)
ax22.errorbar(mG,tC,xerr=mG_u,linestyle='none',c='black',zorder=1,
              elinewidth=0.5)
cbar1=plt.colorbar(sc1,ax=ax22)
cbar1.ax.set_ylabel('Log Erosion Rate [m/Myr]')
plt.xlim((0,0.8))
# plt.xlabel('Mean Gradient [m/m]')
plt.ylim((0,0.8))
plt.legend(loc='best')

[r,p]=pearsonr(e,tC)
lab_str='r = '+str(np.round(r,3))+'; p = '+str(np.round(p,3))
ax23=plt.subplot(9,3,6)
sc1=ax23.scatter(e,tC,s=40,c=ksn,norm=ksn_norm,cmap=ksn_map,zorder=2,
                 label=lab_str)
ax23.errorbar(e,tC,xerr=e_u,linestyle='none',c='black',zorder=1,
              elinewidth=0.5)
cbar1=plt.colorbar(sc1,ax=ax23)
cbar1.ax.set_ylabel('Mean $k_{sn}$ [m]')
plt.xlim((10,10000))
# plt.xlabel('Erosion Rate [m/Myr]')
plt.xscale('log')
plt.ylim((0,0.8))
plt.legend(loc='best')

################
# Mean Elevation
[r,p]=pearsonr(ksn,mnZ)
lab_str='r = '+str(np.round(r,3))+'; p = '+str(np.round(p,3))
ax31=plt.subplot(9,3,7)
sc1=ax31.scatter(ksn,mnZ,s=40,c=np.log10(e),norm=e_norm,cmap=e_map,zorder=2,
                 label=lab_str)
ax31.errorbar(ksn,mnZ,xerr=ksn_u,linestyle='none',c='black',zorder=1,
              elinewidth=0.5)
cbar1=plt.colorbar(sc1,ax=ax31)
cbar1.ax.set_ylabel('Log Erosion Rate [m/Myr]')
plt.xlim((0,550))
# plt.xlabel('Mean $k_{sn}$ [m]')
plt.ylabel('Mean Elevation [m]')
plt.ylim((250,3000))
plt.legend(loc='best')

[r,p]=pearsonr(mG,mnZ)
lab_str='r = '+str(np.round(r,3))+'; p = '+str(np.round(p,3))
ax32=plt.subplot(9,3,8)
sc1=ax32.scatter(mG,mnZ,s=40,c=np.log10(e),norm=e_norm,cmap=e_map,zorder=2,
                 label=lab_str)
ax32.errorbar(mG,mnZ,xerr=mG_u,linestyle='none',c='black',zorder=1,
              elinewidth=0.5)
cbar1=plt.colorbar(sc1,ax=ax32)
cbar1.ax.set_ylabel('Log Erosion Rate [m/Myr]')
plt.xlim((0,0.8))
# plt.xlabel('Mean Gradient [m/m]')
plt.ylim((250,3000))
plt.legend(loc='best')

[r,p]=pearsonr(e,mnZ)
lab_str='r = '+str(np.round(r,3))+'; p = '+str(np.round(p,3))
ax33=plt.subplot(9,3,9)
sc1=ax33.scatter(e,mnZ,s=40,c=ksn,norm=ksn_norm,cmap=ksn_map,zorder=2,
                 label=lab_str)
ax33.errorbar(e,mnZ,xerr=e_u,linestyle='none',c='black',zorder=1,
              elinewidth=0.5)
cbar1=plt.colorbar(sc1,ax=ax33)
cbar1.ax.set_ylabel('Mean $k_{sn}$ [m]')
plt.xlim((10,10000))
# plt.xlabel('Erosion Rate [m/Myr]')
plt.xscale('log')
plt.ylim((250,3000))
plt.legend(loc='best')

################
# Max Elevation
[r,p]=pearsonr(ksn,mxZ)
lab_str='r = '+str(np.round(r,3))+'; p = '+str(np.round(p,3))
ax41=plt.subplot(9,3,10)
sc1=ax41.scatter(ksn,mxZ,s=40,c=np.log10(e),norm=e_norm,cmap=e_map,zorder=2,
                 label=lab_str)
ax41.errorbar(ksn,mxZ,xerr=ksn_u,linestyle='none',c='black',zorder=1,
              elinewidth=0.5)
cbar1=plt.colorbar(sc1,ax=ax41)
cbar1.ax.set_ylabel('Log Erosion Rate [m/Myr]')
plt.xlim((0,550))
# plt.xlabel('Mean $k_{sn}$ [m]')
plt.ylabel('Maximum Elevation [m]')
plt.ylim((1000,4500))
plt.legend(loc='best')

[r,p]=pearsonr(mG,mxZ)
lab_str='r = '+str(np.round(r,3))+'; p = '+str(np.round(p,3))
ax42=plt.subplot(9,3,11)
sc1=ax42.scatter(mG,mxZ,s=40,c=np.log10(e),norm=e_norm,cmap=e_map,zorder=2,
                 label=lab_str)
ax42.errorbar(mG,mxZ,xerr=mG_u,linestyle='none',c='black',zorder=1,
              elinewidth=0.5)
cbar1=plt.colorbar(sc1,ax=ax42)
cbar1.ax.set_ylabel('Log Erosion Rate [m/Myr]')
plt.xlim((0,0.8))
# plt.xlabel('Mean Gradient [m/m]')
plt.ylim((1000,4500))
plt.legend(loc='best')

[r,p]=pearsonr(e,mxZ)
lab_str='r = '+str(np.round(r,3))+'; p = '+str(np.round(p,3))
ax43=plt.subplot(9,3,12)
sc1=ax43.scatter(e,mxZ,s=40,c=ksn,norm=ksn_norm,cmap=ksn_map,zorder=2,
                 label=lab_str)
ax43.errorbar(e,mxZ,xerr=e_u,linestyle='none',c='black',zorder=1,
              elinewidth=0.5)
cbar1=plt.colorbar(sc1,ax=ax43)
cbar1.ax.set_ylabel('Mean $k_{sn}$ [m]')
plt.xlim((10,10000))
# plt.xlabel('Erosion Rate [m/Myr]')
plt.xscale('log')
plt.ylim((1000,4500))
plt.legend(loc='best')

###############
# Mean Rainfall
[r,p]=pearsonr(ksn,mRa)
lab_str='r = '+str(np.round(r,3))+'; p = '+str(np.round(p,3))
ax51=plt.subplot(9,3,13)
sc1=ax51.scatter(ksn,mRa,s=40,c=np.log10(e),norm=e_norm,cmap=e_map,zorder=2,
                 label=lab_str)
ax51.errorbar(ksn,mRa,xerr=ksn_u,linestyle='none',c='black',zorder=1,
              elinewidth=0.5)
cbar1=plt.colorbar(sc1,ax=ax51)
cbar1.ax.set_ylabel('Log Erosion Rate [m/Myr]')
plt.xlim((0,550))
# plt.xlabel('Mean $k_{sn}$ [m]')
plt.ylabel('Mean Rainfall Rate [mm/day]')
plt.ylim((0,5))
plt.legend(loc='best')

[r,p]=pearsonr(mG,mRa)
lab_str='r = '+str(np.round(r,3))+'; p = '+str(np.round(p,3))
ax52=plt.subplot(9,3,14)
sc1=ax52.scatter(mG,mRa,s=40,c=np.log10(e),norm=e_norm,cmap=e_map,zorder=2,
                 label=lab_str)
ax52.errorbar(mG,mRa,xerr=mG_u,linestyle='none',c='black',zorder=1,
              elinewidth=0.5)
cbar1=plt.colorbar(sc1,ax=ax52)
cbar1.ax.set_ylabel('Log Erosion Rate [m/Myr]')
plt.xlim((0,0.8))
# plt.xlabel('Mean Gradient [m/m]')
plt.ylim((0,5))
plt.legend(loc='best')

[r,p]=pearsonr(e,mRa)
lab_str='r = '+str(np.round(r,3))+'; p = '+str(np.round(p,3))
ax53=plt.subplot(9,3,15)
sc1=ax53.scatter(e,mRa,s=40,c=ksn,norm=ksn_norm,cmap=ksn_map,zorder=2,
                 label=lab_str)
ax53.errorbar(e,mRa,xerr=e_u,linestyle='none',c='black',zorder=1,
              elinewidth=0.5)
cbar1=plt.colorbar(sc1,ax=ax53)
cbar1.ax.set_ylabel('Mean $k_{sn}$ [m]')
plt.xlim((10,10000))
# plt.xlabel('Erosion Rate [m/Myr]')
plt.xscale('log')
plt.ylim((0,5))
plt.legend(loc='best')

###############
# Mean Snow 
[r,p]=pearsonr(ksn,mS)
lab_str='r = '+str(np.round(r,3))+'; p = '+str(np.round(p,3))
ax61=plt.subplot(9,3,16)
sc1=ax61.scatter(ksn,mS,s=40,c=np.log10(e),norm=e_norm,cmap=e_map,zorder=2,
                 label=lab_str)
ax61.errorbar(ksn,mS,xerr=ksn_u,linestyle='none',c='black',zorder=1,
              elinewidth=0.5)
cbar1=plt.colorbar(sc1,ax=ax61)
cbar1.ax.set_ylabel('Log Erosion Rate [m/Myr]')
plt.xlim((0,550))
# plt.xlabel('Mean $k_{sn}$ [m]')
plt.ylabel('Mean STD of Mean Snow Cover [%]')
plt.ylim((5,45))
plt.legend(loc='best')

[r,p]=pearsonr(mG,mS)
lab_str='r = '+str(np.round(r,3))+'; p = '+str(np.round(p,3))
ax62=plt.subplot(9,3,17)
sc1=ax62.scatter(mG,mS,s=40,c=np.log10(e),norm=e_norm,cmap=e_map,zorder=2,
                 label=lab_str)
ax62.errorbar(mG,mS,xerr=mG_u,linestyle='none',c='black',zorder=1,
              elinewidth=0.5)
cbar1=plt.colorbar(sc1,ax=ax62)
cbar1.ax.set_ylabel('Log Erosion Rate [m/Myr]')
plt.xlim((0,0.8))
# plt.xlabel('Mean Gradient [m/m]')
plt.ylim((5,45))
plt.legend(loc='best')

[r,p]=pearsonr(e,mS)
lab_str='r = '+str(np.round(r,3))+'; p = '+str(np.round(p,3))
ax63=plt.subplot(9,3,18)
sc1=ax63.scatter(e,mS,s=40,c=ksn,norm=ksn_norm,cmap=ksn_map,zorder=2,
                 label=lab_str)
ax63.errorbar(e,mS,xerr=e_u,linestyle='none',c='black',zorder=1,
              elinewidth=0.5)
cbar1=plt.colorbar(sc1,ax=ax63)
cbar1.ax.set_ylabel('Mean $k_{sn}$ [m]')
plt.xlim((10,10000))
# plt.xlabel('Erosion Rate [m/Myr]')
plt.xscale('log')
plt.ylim((5,45))
plt.legend(loc='best')

###############
# Perc Carb 
[r,p]=pearsonr(ksn,perc_carb)
lab_str='r = '+str(np.round(r,3))+'; p = '+str(np.round(p,3))
ax71=plt.subplot(9,3,19)
sc1=ax71.scatter(ksn,perc_carb,s=40,c=np.log10(e),norm=e_norm,cmap=e_map,zorder=2,
                 label=lab_str)
ax71.errorbar(ksn,perc_carb,xerr=ksn_u,linestyle='none',c='black',zorder=1,
              elinewidth=0.5)
cbar1=plt.colorbar(sc1,ax=ax71)
cbar1.ax.set_ylabel('Log Erosion Rate [m/Myr]')
plt.xlim((0,550))
# plt.xlabel('Mean $k_{sn}$ [m]')
plt.ylabel('Dominant Carbonate [%]')
plt.ylim((-5,105))
plt.legend(loc='best')

[r,p]=pearsonr(mG,perc_carb)
lab_str='r = '+str(np.round(r,3))+'; p = '+str(np.round(p,3))
ax72=plt.subplot(9,3,20)
sc1=ax72.scatter(mG,perc_carb,s=40,c=np.log10(e),norm=e_norm,cmap=e_map,zorder=2,
                 label=lab_str)
ax72.errorbar(mG,perc_carb,xerr=mG_u,linestyle='none',c='black',zorder=1,
              elinewidth=0.5)
cbar1=plt.colorbar(sc1,ax=ax72)
cbar1.ax.set_ylabel('Log Erosion Rate [m/Myr]')
plt.xlim((0,0.8))
# plt.xlabel('Mean Gradient [m/m]')
plt.ylim((-5,105))
plt.legend(loc='best')

[r,p]=pearsonr(e,perc_carb)
lab_str='r = '+str(np.round(r,3))+'; p = '+str(np.round(p,3))
ax73=plt.subplot(9,3,21)
sc1=ax73.scatter(e,perc_carb,s=40,c=ksn,norm=ksn_norm,cmap=ksn_map,zorder=2,
                 label=lab_str)
ax73.errorbar(e,perc_carb,xerr=e_u,linestyle='none',c='black',zorder=1,
              elinewidth=0.5)
cbar1=plt.colorbar(sc1,ax=ax73)
cbar1.ax.set_ylabel('Mean $k_{sn}$ [m]')
plt.xlim((10,10000))
# plt.xlabel('Erosion Rate [m/Myr]')
plt.xscale('log')
plt.ylim((-5,105))
plt.legend(loc='best')

###############
# Perc Clastic
[r,p]=pearsonr(ksn,perc_clastic)
lab_str='r = '+str(np.round(r,3))+'; p = '+str(np.round(p,3))
ax81=plt.subplot(9,3,22)
sc1=ax81.scatter(ksn,perc_clastic,s=40,c=np.log10(e),norm=e_norm,cmap=e_map,zorder=2,
                 label=lab_str)
ax81.errorbar(ksn,perc_clastic,xerr=ksn_u,linestyle='none',c='black',zorder=1,
              elinewidth=0.5)
cbar1=plt.colorbar(sc1,ax=ax81)
cbar1.ax.set_ylabel('Log Erosion Rate [m/Myr]')
plt.xlim((0,550))
# plt.xlabel('Mean $k_{sn}$ [m]')
plt.ylabel('Dominant Clastic [%]')
plt.ylim((-5,105))
plt.legend(loc='best')

[r,p]=pearsonr(mG,perc_clastic)
lab_str='r = '+str(np.round(r,3))+'; p = '+str(np.round(p,3))
ax82=plt.subplot(9,3,23)
sc1=ax82.scatter(mG,perc_clastic,s=40,c=np.log10(e),norm=e_norm,cmap=e_map,zorder=2,
                 label=lab_str)
ax82.errorbar(mG,perc_clastic,xerr=mG_u,linestyle='none',c='black',zorder=1,
              elinewidth=0.5)
cbar1=plt.colorbar(sc1,ax=ax82)
cbar1.ax.set_ylabel('Log Erosion Rate [m/Myr]')
plt.xlim((0,0.8))
# plt.xlabel('Mean Gradient [m/m]')
plt.ylim((-5,105))
plt.legend(loc='best')

[r,p]=pearsonr(e,perc_clastic)
lab_str='r = '+str(np.round(r,3))+'; p = '+str(np.round(p,3))
ax83=plt.subplot(9,3,24)
sc1=ax83.scatter(e,perc_clastic,s=40,c=ksn,norm=ksn_norm,cmap=ksn_map,zorder=2,
                 label=lab_str)
ax83.errorbar(e,perc_clastic,xerr=e_u,linestyle='none',c='black',zorder=1,
              elinewidth=0.5)
cbar1=plt.colorbar(sc1,ax=ax83)
cbar1.ax.set_ylabel('Mean $k_{sn}$ [m]')
plt.xlim((10,10000))
# plt.xlabel('Erosion Rate [m/Myr]')
plt.xscale('log')
plt.ylim((-5,105))
plt.legend(loc='best')

###############
# Perc IgMet
[r,p]=pearsonr(ksn,perc_igmet)
lab_str='r = '+str(np.round(r,3))+'; p = '+str(np.round(p,3))
ax91=plt.subplot(9,3,25)
sc1=ax91.scatter(ksn,perc_igmet,s=40,c=np.log10(e),norm=e_norm,cmap=e_map,zorder=2,
                 label=lab_str)
ax91.errorbar(ksn,perc_igmet,xerr=ksn_u,linestyle='none',c='black',zorder=1,
              elinewidth=0.5)
cbar1=plt.colorbar(sc1,ax=ax91)
cbar1.ax.set_ylabel('Log Erosion Rate [m/Myr]')
plt.xlim((0,550))
plt.xlabel('Mean $k_{sn}$ [m]')
plt.ylabel('Igneous - Metamorphic [%]')
plt.ylim((-5,105))
plt.legend(loc='best')

[r,p]=pearsonr(mG,perc_igmet)
lab_str='r = '+str(np.round(r,3))+'; p = '+str(np.round(p,3))
ax92=plt.subplot(9,3,26)
sc1=ax92.scatter(mG,perc_igmet,s=40,c=np.log10(e),norm=e_norm,cmap=e_map,zorder=2,
                 label=lab_str)
ax92.errorbar(mG,perc_igmet,xerr=mG_u,linestyle='none',c='black',zorder=1,
              elinewidth=0.5)
cbar1=plt.colorbar(sc1,ax=ax92)
cbar1.ax.set_ylabel('Log Erosion Rate [m/Myr]')
plt.xlim((0,0.8))
plt.xlabel('Mean Gradient [m/m]')
plt.ylim((-5,105))
plt.legend(loc='best')

[r,p]=pearsonr(e,perc_igmet)
lab_str='r = '+str(np.round(r,3))+'; p = '+str(np.round(p,3))
ax93=plt.subplot(9,3,27)
sc1=ax93.scatter(e,perc_igmet,s=40,c=ksn,norm=ksn_norm,cmap=ksn_map,zorder=2,
                 label=lab_str)
ax93.errorbar(e,perc_igmet,xerr=e_u,linestyle='none',c='black',zorder=1,
              elinewidth=0.5)
cbar1=plt.colorbar(sc1,ax=ax93)
cbar1.ax.set_ylabel('Mean $k_{sn}$ [m]')
plt.xlim((10,10000))
plt.xlabel('Erosion Rate [m/Myr]')
plt.xscale('log')
plt.ylim((-5,105))
plt.legend(loc='best')




