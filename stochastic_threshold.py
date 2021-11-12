#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Written by Adam M. Forte for 
Low runoff variability driven by a dominance of snowmelt inhibits clear coupling of climate, tectonics, and topography in the Greater Caucasus Mountains

If you use this code or derivatives, please cite the original paper.
"""

import numpy as np
import scipy.integrate as integrate
from scipy.special import gamma
from scipy.special import gammainc
import matplotlib.pyplot as plt
from matplotlib import cm

def set_constants(R,k_e,k_w=15,f=0.08313,omega_a=0.50,
                   omega_s=0.25,alpha=2/3,beta=2/3,a=1.5,tau_c=45,dist_type='inv_gamma'):
    # Convert R in mm/day to k_q in m/s
    k_q=R/(24*60*60*10*100)
    # Derived parameters
    k_t = 0.5*1000*(9.81**(2/3))*(f**(1/3)) # set to 1000 a la Tucker 2004
    y = a*alpha*(1-omega_s) # gamma exponent
    m = a*alpha*(1-omega_a) # m in erosion law
    n = a*beta # n in erosion law
    epsilon = k_e*tau_c**a # threshold term in erosion law
    K = k_e*(k_t**a)*(k_w**(-a*alpha)) #erosional efficiency
    Psi_c=k_e*(tau_c**a)
    con_list=[k_e,k_q,k_t,k_w,f,y,m,n,omega_a,omega_s,alpha,beta,a,
              tau_c,epsilon,K,Psi_c,dist_type]
    return con_list

def unpack_constants(con_list):
    k_e=con_list[0]
    k_q=con_list[1]
    k_t=con_list[2]
    k_w=con_list[3]
    f=con_list[4]
    y=con_list[5]
    m=con_list[6]
    n=con_list[7] 
    omega_a=con_list[8] 
    omega_s=con_list[9]
    alpha=con_list[10]
    beta=con_list[11]
    a=con_list[12]
    tau_c=con_list[13]
    epsilon=con_list[14]
    K=con_list[15]
    Psi_c=con_list[16]
    dist_type=con_list[17]
    return (k_e,k_q,k_t,k_w,f,y,m,n,omega_a,omega_s,alpha,beta,a,
            tau_c,epsilon,K,Psi_c,dist_type)

def an_thresh(E,k,con_list):
    # Convert to meters per second
    I=E/(1e6*365.25*24*60*60)
    [k_e,k_q,k_t,k_w,f,y,m,n,omega_a,
     omega_s,alpha,beta,a,tau_c,epsilon,K,Psi_c,dist_type]=unpack_constants(con_list)
    # Assume return time greater than one day
    p1=(K**(-1/n))*(k_q**(-m/n))
    p2_1=(k+1)*(k+1-y)*gamma(k+1)
    p2_2=(k**(k+1))*y
    p2=(p2_1/p2_2)**(y/(n*(k+1)))
    p3=Psi_c**((k+1-y)/(n*(k+1)))
    p4=I**(y/(n*(k+1)))
    ks=p1*p2*p3*p4
    return ks

def an_const(E,k,con_list):
    # Convert to meters per second
    I=E/(1e6*365.25*24*60*60)
    [k_e,k_q,k_t,k_w,f,y,m,n,omega_a,
     omega_s,alpha,beta,a,tau_c,epsilon,K,Psi_c,dist_type]=unpack_constants(con_list)    
    # Assume return time greater than one day
    p1=K**(-1/n)*k_q**(-m/n)
    p3=(Psi_c+I)**(1/n)
    p2_1=gamma(k+1)*k**(-y)
    p2_2=gamma(k+1-y)
    p2=(p2_1/p2_2)**(1/n)
    ks=p1*p2*p3
    return ks    

def ero_law(Ks,Q_star,con_list):
    [k_e,k_q,k_t,k_w,f,y,m,n,omega_a,
     omega_s,alpha,beta,a,tau_c,epsilon,K,Psi_c,dist_type]=unpack_constants(con_list)    
    # Redefine K
    K=K*(k_q**m)
    # Main Calculation
    E = (1e6*365.25*24*60*60)*(K*(Ks**n)*(Q_star**y)-epsilon) #erosion in m/Ma
    return E

def inv_gamma(Q_star,k):
    pdf=np.exp(-k/Q_star)*((k**(k+1))*Q_star**(-(2+k)))/gamma(k+1)
    return pdf

def wbl(Q_star,k,sc):
    pdf=(k/sc)*((Q_star/sc)**(k-1))*np.exp((-1)*(Q_star/sc)**k)
    return pdf

def ccdf_gamma(Q_star,k):
    cdf=gammainc(k/Q_star,k+1)
    return cdf

def ero_integrand(Ks,Q_star,k,con_list):
    I=ero_law(Ks,Q_star,con_list)
    pdf=inv_gamma(Q_star,k)
    return I*pdf

def ero_integrand_w(Ks,Q_star,k,sc,con_list):
    I=ero_law(Ks,Q_star,con_list)
    pdf=wbl(Q_star,k,sc)
    return I*pdf

def stim_one(Ks,k,con_list,sc=-1):
    [k_e,k_q,k_t,k_w,f,y,m,n,omega_a,
     omega_s,alpha,beta,a,tau_c,epsilon,K,Psi_c,dist_type]=unpack_constants(con_list)    
    # Set integration parameters
    # q_min=0.00368*k
    # q_max=1000000*np.exp(-k)
    q_min=0.001
    q_max=10000
    # Calculate Q_starc
    Q_starc = ((K/epsilon)*(Ks**n)*(k_q**m))**(-1/y)
    if Q_starc < q_min:
        Q_starc = q_min
    elif Q_starc > q_max:
        Q_starc = q_max-1
    if dist_type=='inv_gamma':
        # func=lambda x : ero_integrand(Ks,x,k,con_list)
        # [E,E_err]=integrate.quad(func,Q_starc,q_max)
        x=np.logspace(np.log10(Q_starc),np.log10(q_max),1000)
        y=ero_integrand(Ks,x,k,con_list)
        E=integrate.simpson(y,x)
        E_err=np.zeros(E.shape)
    elif dist_type=='weibull':
        if sc>0:
            # func=lambda x : ero_integrand_w(Ks,x,k,sc,con_list)
            # [E,E_err]=integrate.quad(func,Q_starc,q_max)
            x=np.logspace(np.log10(Q_starc),np.log10(q_max),1000)
            y=ero_integrand_w(Ks,x,k,sc,con_list)
            E=integrate.simpson(y,x)
            E_err=np.zeros(E.shape)
        else:
            raise Exception("Valid argument for the scale parameter must be supplied if using Weibull distribution")
    
    return E,E_err,Q_starc    

def stim_range(k,con_list,sc=-1,max_ksn=700,num_points=200,space_type='log'):
    dist_type=unpack_constants(con_list)[17]
    # Initialize variables
    if space_type=='log':
        Ks=np.logspace(0,np.log10(max_ksn),num=num_points)
    elif space_type=='lin':
        Ks=np.linspace(0,max_ksn,num=num_points)        
    E=np.zeros((num_points,1))
    E_err=np.zeros((num_points,1))
    Q_starc=np.zeros((num_points,1))
    if dist_type=='inv_gamma':
        for i in range(len(Ks)):
            [E[i],E_err[i],Q_starc[i]]=stim_one(Ks[i],k,con_list)
    elif dist_type=='weibull':
        for i in range(len(Ks)):
            [E[i],E_err[i],Q_starc[i]]=stim_one(Ks[i],k,con_list,sc=sc)        
    return Ks,E,E_err,Q_starc

def phi_est(k,con_list):
    [k_e,k_q,k_t,k_w,f,y,m,n,omega_a,
     omega_s,alpha,beta,a,tau_c,epsilon,K,Psi_c]=unpack_constants(con_list)
    phi=(alpha*(1-omega_s))/(beta*(1+k)) 
    return phi

def ret_time(k,E,Ks,con_list):
    [k_e,k_q,k_t,k_w,f,y,m,n,omega_a,
     omega_s,alpha,beta,a,tau_c,epsilon,K,Psi_c]=unpack_constants(con_list)
    [Em,Em_err,Q_starc]=stim_one(Ks,k,con_list)
    tr=(((k+1)*gamma(k+1))/k**(k+1))*Q_starc**(1+k)
    # tr=gammainc(k/Q_starc,k+1)**-1
    # Convert to meters per second
    I=E/(1e6*365.25*24*60*60)
    erat=I/Psi_c
    return [tr,erat]

def plot_all(k,Rb,k_e,k_w=15,f=0.08313,omega_a=0.50,
             omega_s=0.25,alpha=2/3,beta=2/3,a=1.5,tau_c=45,
             max_ksn=700,num_points=200):
    cL=set_constants(Rb,k_e,k_w=k_w,f=f,omega_a=omega_a,
                     omega_s=omega_s,alpha=alpha,beta=beta,
                     a=a,tau_c=tau_c)
    [Ks,E,E_err,Q_starc]=stim_range(k,cL,max_ksn=max_ksn,num_points=num_points)
    Ks_annoT=an_const(E,k,cL)
    Ks_anT=an_thresh(E,k,cL)
    # Plot
    plt.figure(figsize=(10,10))
    plt.plot(E,Ks,c='black',linewidth=2,label='Numerical Solution')
    plt.plot(E,Ks_annoT,c='black',linewidth=2,linestyle='--',label='Constant Discharge')
    plt.plot(E,Ks_anT,c='black',linewidth=2,linestyle=':',label='Analytical with Threshold')
    plt.xlabel('Erosion Rate [m/Myr]')
    plt.ylabel('$k_{sn}$')
    plt.title('k={:.2f}'.format(k)+'; R={:.2f}'.format(Rb)+
              '; $k_e$={:.2e}'.format(k_e)+r'; $\tau_{c}$'+'={:.2f}'.format(tau_c))
    plt.legend(loc='lower right')    
    
def plot_dists(min_k=0.5,max_k=6,k_num=10,lower_Q_star=0.01,upper_Q_star=100):
    Q_star_r=np.logspace(np.log10(lower_Q_star),np.log10(upper_Q_star),100)
    k_r=np.linspace(min_k,max_k,k_num)
        
    plt.figure(num=1,figsize=(10,15))
    ax1=plt.subplot(3,1,1)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('Q*')
    ax1.set_ylabel('Pdf(Q*)')
    
    ax2=plt.subplot(3,1,2)
    ax2.set_xscale('log')
    ax2.set_xlabel('Q*')
    ax2.set_ylabel('Cdf(Q*)')

    ax3=plt.subplot(3,1,3)
    ax3.set_xscale('log')
    ax3.set_yscale('log')
    ax3.set_xlabel('Q*')
    ax3.set_ylabel('N(Q*)') 
   
    cols=cm.get_cmap('plasma',len(k_r))
    
    min_prob=np.zeros(len(k_r))
    for i in range(len(k_r)):
        
        pdf=inv_gamma(Q_star_r,k_r[i])
        cdf=ccdf_gamma(Q_star_r,k_r[i])
        ax1.plot(Q_star_r,pdf,c=np.array(cols(i)))
        ax2.plot(Q_star_r,cdf,c=np.array(cols(i)))
        ax3.plot(Q_star_r,1-cdf,c=np.array(cols(i)),label='k={:2.1f}'.format(k_r[i]))
        min_prob[i]=pdf[-1]

    ax1.set_ylim((np.amin(min_prob),10))
    ax3.legend(loc='best')
    ax3.set_ylim((1/(365*100),1))
    

