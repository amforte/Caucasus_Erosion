#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generates basic plots of erosion rates vs primary topographic metrics,
including normalized channel steepness and gradient. Also an assessment
of whether the choice of reference concavity biases the relationship between
ksn and erosion rate.

Written by Adam M. Forte for 
"Low variability runoff inhibits coupling of climate, tectonics, and 
topography in the Greater Caucasus"

If you use this code or derivatives, please cite the original paper.
"""

import pandas as pd
import matplotlib.pyplot as plt

# Read in main data
df=pd.read_csv('data_tables/gc_ero_master_table.csv')
# Extract main variables of interest
ksn=df['mean_ksn'].to_numpy()
ksn_u=df['se_ksn'].to_numpy()
e=df['St_E_rate_m_Myr'].to_numpy()
e_u=df['St_Ext_Unc'].to_numpy()
mG=df['mean_gradient'].to_numpy()
mG_u=df['se_gradient'].to_numpy()

# Read in ksn at variable concavities
dfc=pd.read_csv('data_tables/ksn_diff_concavities.csv')
ksn0_3=dfc['mean_ksn_0_3'].to_numpy()
ksn0_3_u=dfc['se_ksn_0_3'].to_numpy()
ksn0_4=dfc['mean_ksn_0_4'].to_numpy()
ksn0_4_u=dfc['se_ksn_0_4'].to_numpy()
ksn0_6=dfc['mean_ksn_0_6'].to_numpy()
ksn0_6_u=dfc['se_ksn_0_6'].to_numpy()

# Initiate Figure
fig1=plt.figure(num=1,figsize=(20,15))

plt.subplot(2,3,1)
plt.scatter(e,ksn,s=40,c='k',zorder=2)
plt.errorbar(e,ksn,xerr=e_u,yerr=ksn_u,linestyle='none',c='black',zorder=1,
              elinewidth=0.5)
plt.xlim(10,10000)
plt.ylim(50,550)
plt.xlabel('Erosion Rate [m/Myr]')
plt.ylabel('Mean $k_{sn}$ [m]')
plt.xscale('log')


plt.subplot(2,3,2)
plt.scatter(e,mG,s=40,c='k',zorder=2)
plt.errorbar(e,mG,xerr=e_u,yerr=mG_u,linestyle='none',c='black',zorder=1,
              elinewidth=0.5)
plt.xlim(10,10000)
plt.ylim(0.15,0.75)
plt.xlabel('Erosion Rate [m/Myr]')
plt.ylabel('Mean Gradient [m/m]')
plt.xscale('log')


plt.subplot(2,3,3)
plt.scatter(ksn,mG,s=40,c='k',zorder=2)
plt.errorbar(ksn,mG,xerr=ksn_u,yerr=mG_u,linestyle='none',c='black',zorder=1,
              elinewidth=0.5)
plt.xlim(50,550)
plt.ylim(0.15,0.75)
plt.xlabel('Mean $k_{sn}$ [m]')
plt.ylabel('Mean Gradient [m/m]')

# Ksn at different concavities
plt.subplot(2,3,4)
plt.scatter(e,ksn0_3,s=40,c='k',zorder=2)
plt.errorbar(e,ksn0_3,xerr=e_u,yerr=ksn0_3_u,linestyle='none',c='black',zorder=1,
              elinewidth=0.5)
plt.xlim(10,10000)
plt.ylim(0,25)
plt.xlabel('Erosion Rate [m/Myr]')
plt.ylabel('Mean $k_{sn}$ [$m^{0.6}$]')
plt.xscale('log')
plt.title(r'$\theta_{ref}$=0.3')

plt.subplot(2,3,5)
plt.scatter(e,ksn0_4,s=40,c='k',zorder=2)
plt.errorbar(e,ksn0_4,xerr=e_u,yerr=ksn0_4_u,linestyle='none',c='black',zorder=1,
              elinewidth=0.5)
plt.xlim(10,10000)
plt.ylim(0,110)
plt.xlabel('Erosion Rate [m/Myr]')
plt.ylabel('Mean $k_{sn}$ [$m^{0.8}$]')
plt.xscale('log')
plt.title(r'$\theta_{ref}$=0.4')

plt.subplot(2,3,6)
plt.scatter(e,ksn0_6,s=40,c='k',zorder=2)
plt.errorbar(e,ksn0_6,xerr=e_u,yerr=ksn0_6_u,linestyle='none',c='black',zorder=1,
              elinewidth=0.5)
plt.xlim(10,10000)
plt.ylim(0,2500)
plt.xlabel('Erosion Rate [m/Myr]')
plt.ylabel('Mean $k_{sn}$ [$m^{1.2}$]')
plt.xscale('log')
plt.title(r'$\theta_{ref}$=0.6')