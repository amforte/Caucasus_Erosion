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
from matplotlib import gridspec

import stochastic_threshold as stim

df=pd.read_csv('data_tables/gc_ero_master_table.csv')
# Extract main variables of interest
ksn=df['mean_ksn'].to_numpy()
ksn_u=df['se_ksn'].to_numpy()
e=df['St_E_rate_m_Myr'].to_numpy()
e_u=df['St_Ext_Unc'].to_numpy()
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

# Index
idx=np.where((df['sample_name']!='16AF01') & (df['region']!='LC'))
idx_ngc=np.where((df['sample_name']=='16AF01'))
idx_lc=np.where((df['region']=='LC'))
# Structure error
c_u=np.vstack((clo[idx].reshape(1,len(clo[idx])),chi[idx].reshape(1,len(chi[idx]))))
c_u_ngc=np.vstack((clo[idx_ngc].reshape(1,len(clo[idx_ngc])),chi[idx_ngc].reshape(1,len(chi[idx_ngc]))))
c_u_lc=np.vstack((clo[idx_lc].reshape(1,len(clo[idx_lc])),chi[idx_lc].reshape(1,len(chi[idx_lc]))))


l_u=np.vstack((llo[idx].reshape(1,len(llo[idx])),lhi[idx].reshape(1,len(lhi[idx]))))
l_u_ngc=np.vstack((llo[idx_ngc].reshape(1,len(llo[idx_ngc])),lhi[idx_ngc].reshape(1,len(lhi[idx_ngc]))))
l_u_lc=np.vstack((llo[idx_lc].reshape(1,len(llo[idx_lc])),lhi[idx_lc].reshape(1,len(lhi[idx_lc]))))

g_u=np.vstack((glo[idx].reshape(1,len(glo[idx])),ghi[idx].reshape(1,len(ghi[idx]))))
g_u_ngc=np.vstack((glo[idx_ngc].reshape(1,len(glo[idx_ngc])),ghi[idx_ngc].reshape(1,len(ghi[idx_ngc]))))
g_u_lc=np.vstack((glo[idx_lc].reshape(1,len(glo[idx_lc])),ghi[idx_lc].reshape(1,len(ghi[idx_lc]))))

c_vec=np.linspace(0.1,12,100)
c_vec_l=(c_vec/(10*100))*1e6

dips=np.arange(5,55,10)
z_vec_l=[]
for i in range(len(dips)):
    z_vec=c_vec_l*np.sin(np.deg2rad(dips[i]))
    z_vec_l.append(z_vec)

##### FIGURE 1 ########
fig1=plt.figure(num=1,figsize=(20,6))
dbnorm=colors.Normalize(vmin=0,vmax=100)
ax1=plt.subplot(1,3,1)
sc1=ax1.scatter(l[idx],e[idx],s=60,c=db[idx],norm=dbnorm,cmap='plasma',zorder=2,edgecolors='k')
ax1.errorbar(l[idx],e[idx],xerr=l_u,yerr=e_u[idx],linestyle='none',c='black',zorder=1,elinewidth=0.5)
ax1.scatter(l[idx_ngc],e[idx_ngc],s=60,c=db[idx_ngc],norm=dbnorm,cmap='plasma',zorder=2,edgecolors='k',marker='s')
ax1.errorbar(l[idx_ngc],e[idx_ngc],xerr=l_u_ngc,yerr=e_u[idx_ngc],linestyle='none',c='black',zorder=1,elinewidth=0.5)
ax1.scatter(l[idx_lc],e[idx_lc],s=60,c=db[idx_lc],norm=dbnorm,cmap='plasma',zorder=2,edgecolors='k',marker='^')
ax1.errorbar(l[idx_lc],e[idx_lc],xerr=l_u_lc,yerr=e_u[idx_lc],linestyle='none',c='black',zorder=1,elinewidth=0.5)
cbar1=plt.colorbar(sc1,ax=ax1)
cbar1.ax.set_ylabel('Distance from Center [km]')
ax1.set_xlabel('LC Velocity [mm/yr]')
ax1.set_ylabel('Erosion Rate [m/Myr]')
ax1.set_yscale('log')
ax1.set_ylim((10,10**4))
ax1.set_xlim((0,14))

ax2=plt.subplot(1,3,2)
sc2=ax2.scatter(g[idx],e[idx],s=60,c=db[idx],norm=dbnorm,cmap='plasma',zorder=2,edgecolors='k')
ax2.errorbar(g[idx],e[idx],xerr=g_u,yerr=e_u[idx],linestyle='none',c='black',zorder=1,elinewidth=0.5)
ax2.scatter(g[idx_ngc],e[idx_ngc],s=60,c=db[idx_ngc],norm=dbnorm,cmap='plasma',zorder=2,edgecolors='k',marker='s')
ax2.errorbar(g[idx_ngc],e[idx_ngc],xerr=g_u_ngc,yerr=e_u[idx_ngc],linestyle='none',c='black',zorder=1,elinewidth=0.5)
ax2.scatter(g[idx_lc],e[idx_lc],s=60,c=db[idx_lc],norm=dbnorm,cmap='plasma',zorder=2,edgecolors='k',marker='^')
ax2.errorbar(g[idx_lc],e[idx_lc],xerr=g_u_lc,yerr=e_u[idx_lc],linestyle='none',c='black',zorder=1,elinewidth=0.5)
cbar2=plt.colorbar(sc2,ax=ax2)
cbar2.ax.set_ylabel('Distance from Center [km]')
ax2.set_xlabel('GC Velocity [mm/yr]')
ax2.set_ylabel('Erosion Rate [m/Myr]')
ax2.set_yscale('log')
ax2.set_ylim((10,10**4))
ax2.set_xlim((0,14))

ax3=plt.subplot(1,3,3)
sc3=ax3.scatter(c[idx],e[idx],s=60,c=db[idx],norm=dbnorm,cmap='plasma',zorder=2,edgecolors='k')
ax3.errorbar(c[idx],e[idx],xerr=c_u,yerr=e_u[idx],linestyle='none',c='black',zorder=1,elinewidth=0.5)
ax3.scatter(c[idx_ngc],e[idx_ngc],s=60,c=db[idx_ngc],norm=dbnorm,cmap='plasma',zorder=2,edgecolors='k',marker='s')
ax3.errorbar(c[idx_ngc],e[idx_ngc],xerr=c_u_ngc,yerr=e_u[idx_ngc],linestyle='none',c='black',zorder=1,elinewidth=0.5)
ax3.scatter(c[idx_lc],e[idx_lc],s=60,c=db[idx_lc],norm=dbnorm,cmap='plasma',zorder=2,edgecolors='k',marker='^')
ax3.errorbar(c[idx_lc],e[idx_lc],xerr=c_u_lc,yerr=e_u[idx_lc],linestyle='none',c='black',zorder=1,elinewidth=0.5)
cbar3=plt.colorbar(sc3,ax=ax3)
cbar3.ax.set_ylabel('Distance from Center [km]')
ax3.set_xlabel('LC-GC Convergence [mm/yr]')
ax3.set_ylabel('Erosion Rate [m/Myr]')
ax3.set_yscale('log')
ax3.set_ylim((10,10**4))
ax3.set_xlim((0,14))


fig2=plt.figure(num=2,figsize=(15,8))
ax1=plt.subplot(2,1,1)
dbnorm=colors.Normalize(vmin=0,vmax=100)
sc1=ax1.scatter(c[idx],e[idx],s=60,c=db[idx],norm=dbnorm,cmap='plasma',zorder=3,edgecolors='k')
ax1.errorbar(c[idx],e[idx],xerr=c_u,yerr=e_u[idx],linestyle='none',c='black',zorder=1,elinewidth=0.5)

ax1.scatter(c[idx_ngc],e[idx_ngc],s=60,c=db[idx_ngc],norm=dbnorm,cmap='plasma',zorder=2,edgecolors='k',marker='s')
ax1.errorbar(c[idx_ngc],e[idx_ngc],xerr=c_u_ngc,yerr=e_u[idx_ngc],linestyle='none',c='black',zorder=1,elinewidth=0.5)

ax1.scatter(c[idx_lc],e[idx_lc],s=60,c=db[idx_lc],norm=dbnorm,cmap='plasma',zorder=2,edgecolors='k',marker='^')
ax1.errorbar(c[idx_lc],e[idx_lc],xerr=c_u_lc,yerr=e_u[idx_lc],linestyle='none',c='black',zorder=1,elinewidth=0.5)

for i in range(len(dips)):
    ax1.plot(c_vec,z_vec_l[i],c='k',linestyle='--',linewidth=1,zorder=1)

ax1.plot(c_vec,c_vec_l,c='k',linestyle=':',linewidth=2,zorder=1)

ax1.axhline(300,linestyle=':',zorder=1)
ax1.axhline(500,linestyle=':',zorder=1)

cbar1=plt.colorbar(sc1,ax=ax1)
cbar1.ax.set_ylabel('Distance from Center [km]')
ax1.set_xlabel('GC-LC Convergence [mm/yr]')
ax1.set_ylabel('Erosion Rate [m/Myr]')
ax1.set_yscale('log')
ax1.set_ylim((10,10**4))
ax1.set_xlim((0,12))


ax2=plt.subplot(2,1,2)
cnorm=colors.Normalize(vmin=0,vmax=12)
sc2=ax2.scatter(-db[idx],e[idx],s=60,c=c[idx],norm=cnorm,cmap='Spectral_r',zorder=2,edgecolors='k')
ax2.errorbar(-db[idx],e[idx],yerr=e_u[idx],linestyle='none',c='black',zorder=1,elinewidth=0.5)

ax2.scatter(db[idx_ngc],e[idx_ngc],s=60,c=c[idx_ngc],norm=cnorm,cmap='Spectral_r',zorder=2,edgecolors='k',marker='s')
ax2.errorbar(db[idx_ngc],e[idx_ngc],yerr=e_u[idx_ngc],linestyle='none',c='black',zorder=1,elinewidth=0.5)

ax2.scatter(-db[idx_lc],e[idx_lc],s=60,c=c[idx_lc],norm=cnorm,cmap='Spectral_r',zorder=2,edgecolors='k',marker='^')
ax2.errorbar(-db[idx_lc],e[idx_lc],yerr=e_u[idx_lc],linestyle='none',c='black',zorder=1,elinewidth=0.5)

ax2.axhline(300,linestyle=':',zorder=1)
ax2.axhline(500,linestyle=':',zorder=1)

cbar2=plt.colorbar(sc2,ax=ax2)
cbar2.ax.set_ylabel('GC-LC Convergence')
ax2.set_xlabel('Distance from Center [km]')
ax2.set_ylabel('Erosion Rate [m/Myr]')
ax2.set_yscale('log')
ax2.set_ylim((10,10**4))

#### FIGURE 3
dips=np.arange(5,30,5)
z_vec_l=[]
for i in range(len(dips)):
    z_vec=c_vec_l*np.sin(np.deg2rad(dips[i]))
    z_vec_l.append(z_vec)

gs=gridspec.GridSpec(1,2,width_ratios=[1,2])
fig3=plt.figure(num=3,figsize=(15,6))
ax1=plt.subplot(gs[0])
dbnorm=colors.Normalize(vmin=0,vmax=100)
sc1=ax1.scatter(c[idx],e[idx],s=60,c=db[idx],norm=dbnorm,cmap='plasma',zorder=3,edgecolors='k')
ax1.errorbar(c[idx],e[idx],xerr=c_u,yerr=e_u[idx],linestyle='none',c='black',zorder=1,elinewidth=0.5)

ax1.scatter(c[idx_ngc],e[idx_ngc],s=60,c=db[idx_ngc],norm=dbnorm,cmap='plasma',zorder=2,edgecolors='k',marker='s')
ax1.errorbar(c[idx_ngc],e[idx_ngc],xerr=c_u_ngc,yerr=e_u[idx_ngc],linestyle='none',c='black',zorder=1,elinewidth=0.5)

ax1.scatter(c[idx_lc],e[idx_lc],s=60,c=db[idx_lc],norm=dbnorm,cmap='plasma',zorder=2,edgecolors='k',marker='^')
ax1.errorbar(c[idx_lc],e[idx_lc],xerr=c_u_lc,yerr=e_u[idx_lc],linestyle='none',c='black',zorder=1,elinewidth=0.5)

for i in range(len(dips)):
    ax1.plot(c_vec,z_vec_l[i],c='k',linestyle='--',linewidth=1,zorder=1)

ax1.plot(c_vec,c_vec_l,c='k',linestyle=':',linewidth=2,zorder=1)

cbar1=plt.colorbar(sc1,ax=ax1,orientation='horizontal')
cbar1.ax.set_xlabel('Distance from Center [km]')
ax1.set_xlabel('GC-LC Convergence [mm/yr]')
ax1.set_ylabel('Erosion Rate [m/Myr]')
ax1.set_yscale('log')
ax1.set_ylim((10,10**4))
ax1.set_xlim((0,12))


ax2=plt.subplot(gs[1])
cnorm=colors.Normalize(vmin=2,vmax=10)
sc2=ax2.scatter(-db[idx],e[idx],s=60,c=c[idx],norm=cnorm,cmap='Spectral_r',zorder=2,edgecolors='k')
ax2.errorbar(-db[idx],e[idx],yerr=e_u[idx],linestyle='none',c='black',zorder=1,elinewidth=0.5)

ax2.scatter(db[idx_ngc],e[idx_ngc],s=60,c=c[idx_ngc],norm=cnorm,cmap='Spectral_r',zorder=2,edgecolors='k',marker='s')
ax2.errorbar(db[idx_ngc],e[idx_ngc],yerr=e_u[idx_ngc],linestyle='none',c='black',zorder=1,elinewidth=0.5)

ax2.scatter(-db[idx_lc],e[idx_lc],s=60,c=c[idx_lc],norm=cnorm,cmap='Spectral_r',zorder=2,edgecolors='k',marker='^')
ax2.errorbar(-db[idx_lc],e[idx_lc],yerr=e_u[idx_lc],linestyle='none',c='black',zorder=1,elinewidth=0.5)

cbar2=plt.colorbar(sc2,ax=ax2,orientation='horizontal')
cbar2.ax.set_xlabel('GC-LC Convergence')
ax2.set_xlabel('Distance from Center [km]')
ax2.set_ylabel('Erosion Rate [m/Myr]')
ax2.set_yscale('log')
ax2.set_ylim((10,10**4))



R=1.35
k=4.39
k_e=2.24e-10
cL=stim.set_constants(R,k_e)
[Ks,E,E_err,Q_starc]=stim.stim_range(k,cL,max_ksn=600,num_points=601,space_type='lin')

ksn_vec_l=[]
for i in range(len(dips)):
    # Extract uplift rates
    zv=z_vec_l[i]
    # Iterate through and find closest ksn value to erosion rate
    ks=np.zeros(zv.shape)
    for j in range(len(ks)):
        ix=np.argmin(np.abs(zv[j]-E))
        ks[j]=Ks[ix]
    ksn_vec_l.append(ks)

ksn_vu=np.zeros(c_vec_l.shape)
for i in range(len(c_vec_l)):
    ix=np.argmin(np.abs(c_vec_l[i]-E))
    ksn_vu[i]=Ks[ix]

fig4=plt.figure(num=4,figsize=(15,8))
ax1=plt.subplot(2,1,1)
dbnorm=colors.Normalize(vmin=0,vmax=100)
sc1=ax1.scatter(c[idx],ksn[idx],s=60,c=db[idx],norm=dbnorm,cmap='plasma',zorder=3,edgecolors='k')
ax1.errorbar(c[idx],ksn[idx],xerr=c_u,yerr=ksn_u[idx],linestyle='none',c='black',zorder=1,elinewidth=0.5)

ax1.scatter(c[idx_ngc],ksn[idx_ngc],s=60,c=db[idx_ngc],norm=dbnorm,cmap='plasma',zorder=2,edgecolors='k',marker='s')
ax1.errorbar(c[idx_ngc],ksn[idx_ngc],xerr=c_u_ngc,yerr=ksn_u[idx_ngc],linestyle='none',c='black',zorder=1,elinewidth=0.5)

ax1.scatter(c[idx_lc],ksn[idx_lc],s=60,c=db[idx_lc],norm=dbnorm,cmap='plasma',zorder=2,edgecolors='k',marker='^')
ax1.errorbar(c[idx_lc],ksn[idx_lc],xerr=c_u_lc,yerr=ksn_u[idx_lc],linestyle='none',c='black',zorder=1,elinewidth=0.5)

for i in range(len(dips)):
    ax1.plot(c_vec,ksn_vec_l[i],c='k',linestyle='--',linewidth=1,zorder=1)
ax1.plot(c_vec,ksn_vu,c='k',linestyle=':',linewidth=3,zorder=1)

cbar1=plt.colorbar(sc1,ax=ax1)
cbar1.ax.set_ylabel('Distance from Center [km]')
ax1.set_xlabel('GC-LC Convergence [mm/yr]')
ax1.set_ylabel('$k_{sn}$ [m]')
ax1.set_xlim((0,12))
ax1.set_ylim((0,550))


ax2=plt.subplot(2,1,2)
cnorm=colors.Normalize(vmin=0,vmax=12)
sc2=ax2.scatter(-db[idx],ksn[idx],s=60,c=c[idx],norm=cnorm,cmap='Spectral_r',zorder=2,edgecolors='k')
ax2.errorbar(-db[idx],ksn[idx],yerr=ksn_u[idx],linestyle='none',c='black',zorder=1,elinewidth=0.5)

ax2.scatter(db[idx_ngc],ksn[idx_ngc],s=60,c=c[idx_ngc],norm=cnorm,cmap='Spectral_r',zorder=2,edgecolors='k',marker='s')
ax2.errorbar(db[idx_ngc],ksn[idx_ngc],yerr=ksn_u[idx_ngc],linestyle='none',c='black',zorder=1,elinewidth=0.5)

ax2.scatter(-db[idx_lc],ksn[idx_lc],s=60,c=c[idx_lc],norm=cnorm,cmap='Spectral_r',zorder=2,edgecolors='k',marker='^')
ax2.errorbar(-db[idx_lc],ksn[idx_lc],yerr=ksn_u[idx_lc],linestyle='none',c='black',zorder=1,elinewidth=0.5)

cbar2=plt.colorbar(sc2,ax=ax2)
cbar2.ax.set_ylabel('GC-LC Convergence')
ax2.set_xlabel('Distance from Center [km]')
ax2.set_ylabel('$k_{sn}$ [m]')
ax2.set_ylim((0,550))



fig5=plt.figure(num=5,figsize=(10,10))
ax1=plt.subplot()
plt.scatter(e,ksn,s=50,c='k',zorder=2)
plt.errorbar(e,ksn,yerr=ksn_u,xerr=e_u,linestyle='none',c='black',zorder=1,elinewidth=0.5)
plt.xscale('log')
plt.yscale('log')
ax1.axvline(300,linestyle=':')
ax1.axvline(500,linestyle=':')
plt.xlabel('Erosion Rate [m/Myr]')
plt.ylabel('$k_{sn}$ [m]')
