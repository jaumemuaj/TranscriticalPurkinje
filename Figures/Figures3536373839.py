#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 18:44:53 2025

@author: jaume
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Ellipse
import matplotlib.transforms as transforms
from TCplusplus import *


#THIS SCRIPT GENERATE PHASE SPACE DIAGRAMS FROM A VERY LONG TRACE WHICH CONTAINS 
#ALL 5 PHENOMENOLOGIES.
#NEEDS TO BE EDITED IF YOU WANT TO GENERATE PLOTS FROM A DIFFERENT TRACE
#MAINLY THE TIME INTERVALS PLOTTED
#STILL, THIS IS A GOOD TEMPLATE TO USE AS HELP

tr=np.load('trace_5phen.npz')
time=tr['t']/ms - 100


# Define the parameters
a = 0.1  # Soma voltage influence on adaptation variable dynamics
b = -3   # Coefficient for fitting w*v term
vreset = -55  # Reset potential
wreset = 15   # Reset adaptation
eps = 1  # Scaling adaptation dynamics
Iint = 90  # Current to fire at PC intrinsic freq (representing sodium currents)

C = 5  # Capacitance soma (unit nF)

v_sp = -5  # Threshold potential soma
eps_z = 0.1  # Scaling pump dynamics
d_z = 40  # Pumping addition on spikes

el = -61  # Resting potential

tauw = 1  # Adaptation time constant
w0 = 4
g = 3  # Coupling coefficient

def TCO_nullclines(zs, ax, Vd_values,vs_labels,w0s):
    
    Vs = np.linspace(-100, 50, 10000)

    for z, col,label in zip(zs, ['black', 'dimgray', 'darkgray'],vs_labels):
        for Vd in Vd_values:
            # Define the range for Vs
            # Define the ws nullcline
            # Define the Vs nullcline by solving for ws
            def Vs_nullcline(V, el, b, I, z, g, Vd):
                coefs = [1, -b*(V - el), -(V - el)**2 - I + z - g * (Vd-V)]
                roots = np.roots(coefs)
                real_roots = roots[np.isreal(roots)].real
                if len(real_roots) < 2:
                    real_roots = np.pad(real_roots, (0, 2 - len(real_roots)), constant_values=np.nan)
                return real_roots

            # Calculate the nullclines and separate branches
            Vs_nullcline_ws = np.array([Vs_nullcline(V, el, b, Iint, z, g, Vd) for V in Vs])

            # Separate the two branches of the Vs nullcline, avoiding the bridge
            branch_1_Vs = []
            branch_1_ws = []
            branch_2_Vs = []
            branch_2_ws = []

            for i in range(len(Vs)):
                if len(Vs_nullcline_ws[i]) == 2:
                    ws_1, ws_2 = Vs_nullcline_ws[i]
                    if ws_1 < ws_2:
                        branch_1_Vs.append(Vs[i])
                        branch_1_ws.append(ws_1)
                        branch_2_Vs.append(Vs[i])
                        branch_2_ws.append(ws_2)
                    else:
                        branch_1_Vs.append(Vs[i])
                        branch_1_ws.append(ws_2)
                        branch_2_Vs.append(Vs[i])
                        branch_2_ws.append(ws_1)
                else:
                    branch_1_Vs.append(np.nan)
                    branch_1_ws.append(np.nan)
                    branch_2_Vs.append(np.nan)
                    branch_2_ws.append(np.nan)

            # Convert to numpy arrays for easier plotting
            branch_1_Vs = np.array(branch_1_Vs)
            branch_1_ws = np.array(branch_1_ws)
            branch_2_Vs = np.array(branch_2_Vs)
            branch_2_ws = np.array(branch_2_ws)

            # Plot the nullclines
            b1, = ax.plot(branch_1_Vs, branch_1_ws, color=col)
            b2, = ax.plot(branch_2_Vs, branch_2_ws, color=b1.get_color(), label=f'$V_s$ NC, {label}')
    
    for w0,linestyle in zip(w0s,['--','dotted','dashdot']):
        ws_nullcline = a * (Vs - el) + w0
        ax.plot(Vs, ws_nullcline, label=r'$w_s$ NC, $w_0-\ \alpha[Ca^{2+}]$ ='+f' {w0}', color='black', linestyle=linestyle)

    #w0=0
    #ax.plot(Vs, a * (Vs - el) + w0, label=f'$w_s$ nullcline, w0={w0}', color='black', linestyle='--')
    #w0=4
    #ax.plot(Vs, a * (Vs - el) + w0, label=f'$w_s$ nullcline, w0={w0}', linestyle='--')
    #w0=-4
    #ax.plot(Vs, a * (Vs - el) + w0, label=f'$w_s$ nullcline, w0={w0}', linestyle='--', color='tomato')
    white_box = Rectangle((-55.5,14.6), 2, 1, linewidth=1, edgecolor='black', facecolor='white',zorder=5)
    ax.add_patch(white_box)

def segment_trajectory(trajectory, time, jump_indices,start):
    
    jump_indices=jump_indices-start
    print(jump_indices)
    segments = []
    start_idx = 0
    dt=0.02
    time=time/ms
    for idx in jump_indices:
        #print(start_idx,idx)
        segment = (time[int(start_idx):int(idx/dt)], trajectory[int(start_idx):int(idx/dt)])
        segments.append(segment)
        start_idx = int(idx/dt) +2  # Skip the jump index itself
        
    #print(idx)
    # Add the last segment
    if start_idx < len(trajectory):
        segment = (time[start_idx:], trajectory[start_idx:])
        segments.append(segment)

    return segments


#INTRINSIC FIRING


start,stop=[100,200]
dt=0.02
# Example usage
fig, (ax,ax1) = plt.subplots(1,2,figsize=(10,3.5))
fig.subplots_adjust(wspace=0.3)

TCO_nullclines([-10,45], ax, [-55], ['depolarization phase', 'repolarization phase'],w0s=[4])
ax.legend(loc='upper right',fontsize=10)
ax.set_xlim((-80,0))
ax.set_ylim((-5,20))


spikes=tr['spikes']
sv=segment_trajectory(tr['Soma'][int(start/dt):int(stop/dt)],tr['t'][int(start/dt):int(stop/dt)],spikes[(spikes>start)&(spikes<stop)],start=start)
sw=segment_trajectory(tr['ws'][int(start/dt):int(stop/dt)],tr['t'][int(start/dt):int(stop/dt)],spikes[(spikes>start)&(spikes<stop)],start=start)

#print(len(sv[1][1]),len(sw))
#print(sv[1][1])
for v,w in zip(sv,sw):
    v=v[1]
    w=w[1]
    ax.plot(v/mV, w/nA, color='tomato')
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlabel(r'$V_s$')
ax.set_ylabel(r'$w_s$')
ax.set_ylim((0,20))

ax1.plot(time,tr['Soma']/mV,color='tomato')
ax1.set_xlim((start-100,stop-100))
ax1.set_yticks([-60,-35,-10])
ax1.set_xticks([0,50,100])
ax1.set_xlabel('ms')
ax1.set_ylabel('mV')
ax1.set_ylim((-75,0))
plt.tight_layout()
plt.show()


#%%
#inhibition

start,stop=[239.55,310]
dt=0.02
Iint=75
# Example usage
fig, (ax,ax1) = plt.subplots(1,2,figsize=(10,3.5))
fig.subplots_adjust(wspace=0.3)

TCO_nullclines([0,-30], ax, [-55], ['inhibition phase','depolarization phase'],w0s=[4])
ax.legend(loc='upper right',fontsize=10)
ax.set_xlim((-80,0))
ax.set_ylim((-5,20))


spikes=tr['spikes']
sv=segment_trajectory(tr['Soma'][int(start/dt):int(stop/dt)],tr['t'][int(start/dt):int(stop/dt)],spikes[(spikes>start)&(spikes<stop)],start=start)
sw=segment_trajectory(tr['ws'][int(start/dt):int(stop/dt)],tr['t'][int(start/dt):int(stop/dt)],spikes[(spikes>start)&(spikes<stop)],start=start)

#print(len(sv[1][1]),len(sw))
#print(sv[1][1])
for v,w in zip(sv,sw):
    v=v[1]
    w=w[1]
    ax.plot(v[:int(50/dt)]/mV, w[:int(50/dt)]/nA, color='tomato')
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlabel(r'$V_s$')
ax.set_ylabel(r'$w_s$')
ax.set_ylim((0,20))

v=sv[0][1]
w=sw[0][1]
print(v,w)
ax.plot(v[int((300-start)/dt):]/mV,w[int((300-start)/dt):]/nA,linestyle='--',color='tomato')


ax1.plot(time[int(300/dt):],tr['Soma'][int(300/dt):]/mV,color='tomato',linestyle='--')
ax1.plot(time[:int(300/dt)],tr['Soma'][:int(300/dt)]/mV,color='tomato')
ax1.set_yticks([-60,-35,-10])
ax1.set_xticks([start-100,start+(stop-start)/2-100,stop-100],[0,35,70])
ax1.set_xlabel('ms')
ax1.set_ylabel('mV')
ax1.set_ylim((-75,0))

ax1.set_xlim((start-100,stop-100))
plt.tight_layout()
plt.show()
#%%
#PF no pause

start,stop=[390,460]
dt=0.02
Iint=91
zetas=[65,10]
vds=[-52.5]
w0=3.75
labels=['repolarization phase','depolarization phase']

startstim,stopstim=(400,450)

# Example usage
fig, (ax,ax1) = plt.subplots(1,2,figsize=(10,3.5))
fig.subplots_adjust(wspace=0.3)


TCO_nullclines(zetas, ax, vds, labels,w0s=[4,3.75])
ax.legend(loc='upper right',fontsize=10)
ax.set_xlim((-80,0))
ax.set_ylim((-5,20))


spikes=tr['spikes']
svs=segment_trajectory(tr['Soma'][int(start/dt):int(stop/dt)],tr['t'][int(start/dt):int(stop/dt)],spikes[(spikes>start)&(spikes<stop)],start=start)
sws=segment_trajectory(tr['ws'][int(start/dt):int(stop/dt)],tr['t'][int(start/dt):int(stop/dt)],spikes[(spikes>start)&(spikes<stop)],start=start)

#print(len(sv[1][1]),len(sw))
#print(sv[1][1])
for sv,sw in zip(svs,sws):
    v=sv[1]
    w=sw[1]
    print(sv[0][0])
    if sv[0][0]<startstim or sv[0][0]>stopstim:
        ax.plot(v/mV, w/nA, color='tomato',linestyle='--')
    else:
        ax.plot(v/mV, w/nA, color='tomato')
v=sv[1][0]
w=sw[0][1]
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlabel(r'$V_s$')
ax.set_ylabel(r'$w_s$')
ax.set_ylim((0,20))
#print(v[0])

#print(v,w)
#ax.plot(v/mV,w/nA,linestyle='--',color='tomato')

ax1.plot(time,tr['Soma']/mV,color='tomato',linestyle='--')
ax1.plot(time[int(startstim/dt):int(stopstim/dt)],tr['Soma'][int(startstim/dt):int(stopstim/dt)]/mV,color='tomato')
ax1.set_yticks([-60,-35,-10])
ax1.set_xticks([start-100,start+(stop-start)/2-100,stop-100],[0,35,70])
ax1.set_xlabel('ms')
ax1.set_ylabel('mV')
ax1.set_ylim((-75,0))
ax1.set_xlim((start-100,stop-100))
plt.tight_layout()
plt.show()

#%%
#PF with pause

start,stop=[690,760]
dt=0.02
Iint=91
zetas=[120,65,10]
vds=[-52]
labels=['pausing phase','repolarization phase','depolarization phase']
w0s=[4,3.65]

startstim,stopstim=(700,750)

# Example usage
# Example usage
fig, (ax,ax1) = plt.subplots(1,2,figsize=(10,3.5))
fig.subplots_adjust(wspace=0.3)

TCO_nullclines(zetas, ax, vds, labels,w0s=w0s)
ax.legend(loc='upper right',fontsize=10)
ax.set_xlim((-80,0))
ax.set_ylim((-5,20))


spikes=tr['spikes']
svs=segment_trajectory(tr['Soma'][int(start/dt):int(stop/dt)],tr['t'][int(start/dt):int(stop/dt)],spikes[(spikes>start)&(spikes<stop)],start=start)
sws=segment_trajectory(tr['ws'][int(start/dt):int(stop/dt)],tr['t'][int(start/dt):int(stop/dt)],spikes[(spikes>start)&(spikes<stop)],start=start)

#print(len(sv[1][1]),len(sw))
#print(sv[1][1])
for sv,sw in zip(svs,sws):
    v=sv[1]
    w=sw[1]
    print(sv[0][0])
    if sv[0][0]<startstim or sv[0][0]>stopstim:
        ax.plot(v/mV, w/nA, color='tomato',linestyle='--')
    else:
        ax.plot(v/mV, w/nA, color='tomato')
v=sv[1][0]
w=sw[0][1]
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlabel(r'$V_s$')
ax.set_ylabel(r'$w_s$')
#print(v[0])

#print(v,w)
#ax.plot(v/mV,w/nA,linestyle='--',color='tomato')

ax1.plot(time,tr['Soma']/mV,color='tomato',linestyle='--')
ax1.plot(time[int(startstim/dt):int(stopstim/dt)],tr['Soma'][int(startstim/dt):int(stopstim/dt)]/mV,color='tomato')
ax1.set_yticks([-60,-35,-10])
ax1.set_xticks([start-100,start+(stop-start)/2-100,stop-100],[0,35,70])
ax1.set_xlabel('ms')
ax1.set_ylabel('mV')
ax1.set_ylim((-75,0))
ax1.set_xlim((start-100,stop-100))
plt.tight_layout()
plt.show()

#%%
#CF
start,stop=[856,1000]
dt=0.02
Iint=91
zetas=[120,85,20]
vds=[-55]
labels=['pausing phase','burst phase','depolarization phase']
w0s=[3.7,0,-3.5]

startstim,stopstim=(850,900)

# Example usage
# Example usage
fig, (ax,ax1) = plt.subplots(1,2,figsize=(10,4))

TCO_nullclines(zetas, ax, vds, labels,w0s=w0s)
ax.legend(loc='upper right',fontsize=10)
ax.set_xlim((-80,0))
ax.set_ylim((-10,20))


spikes=tr['spikes']
svs=segment_trajectory(tr['Soma'][int(start/dt):int(stop/dt)],tr['t'][int(start/dt):int(stop/dt)],spikes[(spikes>start)&(spikes<stop)],start=start)
sws=segment_trajectory(tr['ws'][int(start/dt):int(stop/dt)],tr['t'][int(start/dt):int(stop/dt)],spikes[(spikes>start)&(spikes<stop)],start=start)

#print(len(sv[1][1]),len(sw))
#print(sv[1][1])
for sv,sw in zip(svs,sws):
    v=sv[1]
    w=sw[1]
    print(sv[0][0])
    if sv[0][-10]<start+15.64:
        ax.plot(v/mV, w/nA, color='tomato',linestyle='-')
    else:
        ax.plot(v/mV, w/nA, color='darkred',linestyle='-')

v=sv[1][0]
w=sw[0][1]
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlabel(r'$V_s$')
ax.set_ylabel(r'$w_s$')
#print(v[0])

#print(v,w)
#ax.plot(v/mV,w/nA,linestyle='--',color='tomato')

ax1.plot(time,tr['Soma']/mV,color='tomato',linestyle='--')
ax1.plot(time[int(startstim/dt):int(862/dt)],tr['Soma'][int(startstim/dt):int(862/dt)]/mV,color='tomato',label='Complex Spike')
ax1.plot(time[int(860/dt):int(1000/dt)],tr['Soma'][int(860/dt):int(1000/dt)]/mV,color='darkred',label='Pause')
ax1.legend(fontsize=10)
ax1.set_yticks([-60,-35,-10])
ax1.set_xticks([start-100-10,start-10+(stop-start+10)/2-100,stop-100],[0,75,150])
ax1.set_xlabel('ms')
ax1.set_ylabel('mV')
ax1.set_ylim((-80,0))
ax1.set_xlim((start-100-10,stop-100))


#ax1.set_xlim((start-100-10,stop-100))
plt.tight_layout()
plt.show()