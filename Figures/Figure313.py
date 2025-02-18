#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 17:13:40 2025

@author: jaume
"""

from TCplusplus import *

import time
import matplotlib.pyplot as plt
from functools import partial
import typing
import jax
import jax.numpy as jnp
import seaborn as sns
import json
import numpy as np
from scipy.ndimage import gaussian_filter1d
 

exp = jnp.exp
uS = 1e-6; mV = 1e-3; nA = 1e-9; nF = 1e-9; ms = 1e-3


def plot_sorted_raster(ax,spike_times, neuron_values,yticks=False):
    # Sort the neurons based on neuron_values in descending order
    sorted_indices = np.argsort(neuron_values)[::1]
    # Create a new sorted spike_times array
    sorted_spike_times = [spike_times[i] for i in sorted_indices]
    
    # Iterate through each neuron and its spike times
    for neuron_id, neuron_spike_times in enumerate(sorted_spike_times):
        ax.vlines(neuron_spike_times, neuron_id + 0.5, neuron_id + 1.5)
    
    
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Neuron')
    #plt.ylim(0.5, len(spike_times) + 0.5)
    if yticks==True:
        ax.set_yticks(np.linspace(1, len(spike_times) + 1,3), 
                   np.round(np.array([min(neuron_values),round(np.mean(neuron_values),2),max(neuron_values)]),1))

def smooth_freq(spike_times, bin_size=1, start_time=0, end_time=10, sigma=5, plot=True):
    """
    Plot the smoothed firing rates of neurons over time.

    Parameters:
    - spike_times: list of lists, each containing spike times for a neuron
    - bin_size: size of the time bins in ms
    - start_time: start time for the analysis in ms
    - end_time: end time for the analysis in ms
    - sigma: standard deviation for Gaussian kernel in ms
    """
    # Convert bin size to seconds for Hz calculation
    bin_size_sec = bin_size / 1000.0

    # Create bins
    bins = np.arange(start_time, end_time + bin_size, bin_size)

    # Calculate the firing rate for each neuron
    firing_rates = []
    for spikes in spike_times:
        counts, _ = np.histogram(spikes, bins)
        firing_rate = counts / bin_size_sec  # Convert to rate (spikes/sec)
        smoothed_firing_rate = gaussian_filter1d(firing_rate, sigma=sigma / bin_size)
        firing_rates.append(smoothed_firing_rate)

    # Convert to a numpy array for easier manipulation
    firing_rates = np.array(firing_rates)

    # Calculate the mean firing rate across all neurons
    mean_freq_trace = np.mean(firing_rates, axis=0)
    
    
    id_freq_avg=np.array([np.mean(f) for f in firing_rates])
    
    
    # Plot the firing rates and the mean firing rate
    
    time_bins = bins[:-1] + bin_size / 2  # Center of each bin

    
    if plot==True:
        plt.figure(figsize=(10, 6))
        
        # Plot each neuron's firing rate
        for i, fr in enumerate(firing_rates):
            plt.plot(time_bins, fr, alpha=0.5)
    
        # Highlight the mean firing rate
        plt.plot(time_bins, mean_freq_trace, label='Mean', color='black', linewidth=2)
    
        # Beautify the plot
        plt.xlabel('Time (ms)')
        plt.ylabel('Firing Rate (Hz)')
        plt.title('Smoothed Firing Rates of Neurons Over Time')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
    
        # Show the plot
        plt.show()
        
    return mean_freq_trace, id_freq_avg, time_bins, firing_rates  




#MAIN 2 gives population variability, batches of 1
def main2():
    nsteps = 50000 # 50000 steps = 1 second
    dt = 0.02*ms
    t = dt * jnp.arange(nsteps)
    N=100
    pct=0.05   
    variation=['Iint','zmax','Deltaeps_z','max_vCas','dres_coeff','m','tauR','Ca_half','slp','tauCa','C',
          'Cd','Csd','max_vCad','Kpf','Kcf','vths','c','p']
    spike_raster=[]
    pauselist=[]
    avg_freqs=[]
    std_freqs=[]

    Is=[0]
    for idx,Iinj in enumerate(Is):
        for i in range(N):
            with open('params1606.json', 'r') as file:
                params=json.load(file)
            var=pct*(np.round(np.random.rand(len(variation)),3)*2-1)
            new_params={}
            for param,v in zip(variation,var):
                params[param]+=v*params[param]
                new_params[param]=params[param]
                
            calibration=200*ms   
            Iexc = np.zeros(nsteps)
            IinjnA = Iinj * nA
            Iexc[int(calibration/dt):]=IinjnA
            Iexc=jnp.array(Iexc)
            
                    
            tcf=510
            stims = Stim(
                Ipf1=jnp.zeros(nsteps),
                Ipf2=jnp.zeros(nsteps),
                Ipf3=jnp.zeros(nsteps),
                Icf=10 * nA * ((t % 1 > tcf * ms) & (t % 1 < (tcf+5) * ms)),
                #Icf=jnp.zeros(nsteps),
                Iexc=Iexc,
                Iinh=jnp.zeros(nsteps)
            )
            
            #params['Deltaeps_z']=5
            #params['dres_coeff']=3
            #params['zmax']=150*nA
            params = Params.init(params)
            state0 = State.init(params)
            
            a = time.time()
            _, trace = jax.lax.scan(
                lambda state, stim: timestep(params, state, stim, dt),
                state0, stims
            )
            
            b = time.time()
            
            spike_indices = jnp.where(trace.sspike)[0]
            spike_times = spike_indices * dt *1e3
            freq=jnp.round(jnp.array([1/(spike_times[i+1]-spike_times[i]) for i in range(len(spike_times)-1)])*1000,2)
            
                
            pauses=jnp.round(jnp.array([(spike_times[i+1]-spike_times[i]) for i in range(len(spike_times)-1)]),2)
            spike_raster.append(spike_times)
            pauselist.append(np.max(pauses))
            print('simulation took:', b - a, 'seconds')

        batch=spike_raster[N*idx:N*(idx+1)]
        _, id_freq_avg,_,_=smooth_freq(batch, bin_size=15, start_time=200, end_time=500, sigma=30,plot=False)
        avg_freqs.append(np.mean(id_freq_avg))
        std_freqs.append(np.std(id_freq_avg))
    return spike_raster,pauselist,Is,avg_freqs,std_freqs

s,p,Is,f,std=main2()


#BIG PLOT
import seaborn as sns
from scipy.stats import norm

sns.set(style="ticks", context="paper", font_scale=1.5)

s_s=s[200:400]
p_s=p[200:400]

fig = plt.figure(figsize=(10, 12))

gs= plt.GridSpec(2,2,height_ratios=[3,2],width_ratios=[1,1])
gs.update(wspace=0.3)
gs.update(hspace=0.3)


ax1=plt.subplot(gs[0,0])
plot_sorted_raster(ax1, s, p)
ax1.set_ylabel('Neuron nº',fontsize=12)
ax1.set_xlabel('Time(ms)',fontsize=12)
ax1.set_xticks([100,300,500,700,900],[-400,-200,0,200,400])

ax2=plt.subplot(gs[0,1])

n, bins, patches = ax2.hist(p, bins=30, color='skyblue', edgecolor='black', density=True)

# Fit a normal distribution to the data: mean and standard deviation
mu, std = norm.fit(p)

# Plot the Gaussian fit
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
fit = norm.pdf(x, mu, std)
ax2.plot(x, fit, 'k', linewidth=2, label='Gaussian fit')

# Title and labels
ax2.set_xlabel('Pause length (ms)', fontsize=12)
ax2.set_ylabel('')  # Remove y-axis label
ax2.set_yticks([])  # Remove y-axis ticks
ax2.legend(loc='upper right',fontsize=10)

ax3=plt.subplot(gs[1,:])
mean_freq_trace,id_freq_avg,t,frates=smooth_freq(s, bin_size=15, start_time=200, end_time=900, sigma=30,plot=False)
print(np.mean(p))
for f in frates:
    ax3.plot(t,f,alpha=0.5)
ax3.plot(t,mean_freq_trace,color='black',linewidth=2, label='Mean frequency trace')
ax3.set_ylabel('Frequency (Hz)',fontsize=12)
ax3.set_xlabel('Time (ms)',fontsize=12)
ax3.legend(loc='upper right', fontsize=10)
ax3.set_xticks([100,300,400,500,600,700,800],[-400,-200,-100,0,100,200,300])
ax3.set_xlim((250,850))


plt.tight_layout()
#plt.savefig('population.png',dpi=300)
plt.show()
