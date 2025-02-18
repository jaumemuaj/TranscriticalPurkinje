#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 17:45:17 2025

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


def main3(batch_no=10,current_no=30,cf=0):
    nsteps = 50000 # 50000 steps = 1 second
    dt = 0.02*ms
    t = dt * jnp.arange(nsteps)
    N=batch_no
    pct=0.05
    variation=['Iint','zmax','Deltaeps_z','max_vCas','dres_coeff','m','tauR','Ca_half','slp','tauCa','C',
          'Cd','Csd','max_vCad','Kpf','Kcf','vths','c','p','d_z']
    
    spike_raster=[]
    pauselist=[]
    avg_freqs=[]
    std_freqs=[]
    avg_pauses=[]
    std_pauses=[]
    
    steps=30 
    
    Is=np.linspace(-10,35,current_no)
    
    trial=0
    for idx,Iinj in enumerate(Is):
        for i in range(N):
            with open('params1606.json', 'r') as file:
                params=json.load(file)
                params['Iint']=92e-9+Iinj*1e-9
                
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
            
                    
            tcf=500
            tpf=300
            stims = Stim(
                Ipf1=jnp.zeros(nsteps),
                Ipf2=jnp.zeros(nsteps),
                Ipf3=jnp.zeros(nsteps),
                Icf=cf * nA * ((t % 1 > tcf * ms) & (t % 1 < (tcf+5) * ms)),
                #Icf=jnp.zeros(nsteps),
                Iexc=0*Iexc,
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
            spike_raster.append(np.array(spike_times))
            pauselist.append(np.max(pauses))
            print('simulation took:', b - a, 'seconds')
            
            
        start,stop=[int(steps*trial+N*idx),int(steps*trial+N*(idx+1))]
        batch=spike_raster[start:stop]
        avg_pause=np.mean(pauselist[start:stop])
        std_pause=np.std(pauselist[start:stop])
        _, id_freq_avg,_,_=smooth_freq(batch, bin_size=15, start_time=200, end_time=500, sigma=30,plot=False)
        
        avg_freqs.append(np.mean(id_freq_avg))
        std_freqs.append(np.std(id_freq_avg))
        avg_pauses.append(avg_pause)
        std_pauses.append(std_pause)
    
    output={}
    output['spikeraster']=spike_raster;output['pauselist']=pauselist;output['Iinj']=Is
    output['avgfreqs']=avg_freqs;output['stdfreqs']=std_freqs;output['avgpauses']=avg_pauses
    output['stdpauses']=std_pauses
        
    return output

fraster=main3(batch_no=1,current_no=100,cf=10)
fp=main3(cf=10)

#To analyze another set of data, use MAIN3 function to create batches of N
#PCs with a certain current injection. Fit the resulting data with curve_fit from scipy.optimize
# Figure 3.14, right

#The raster plot is made from represantatives of each current injection, to visualize the
#relationship between firing frequency and CF pause duration



#THIS is the original dataset: if used, please check the labels of the .npz datsets inside
#as the labels are slightly different.

#fp=np.load('fp.npz')
#fraster=np.load('fraster.npz')

#%%
f=fraster['avgfreqs']
f1=np.array(fp['avgfreqs'])
pavg=np.array(fp['avgpauses'])
pstd=np.array(fp['stdpauses'])
sorted_indices = np.argsort(f)[::1]
sorted_spike_times = [fraster['spikeraster'][i] for i in sorted_indices]
#%%
from scipy.optimize import curve_fit

sns.set(style="ticks", context="paper", font_scale=1.5)


# Select a fraction of the points (e.g., 10%)
# fraction = 0.5
# indices = np.random.choice(len(f1), size=int(len(f1) * fraction), replace=False).astype(int)
# indices=np.sort(indices)
# f_subsampled = f1[indices]
# pavg_subsampled = pavg[indices]
# pstd_subsampled = pstd[indices]

sns.set(style="ticks", context="paper", font_scale=1.5)

# Define a linear function for fitting
def linear_func(x, b, c):
    return  b * x + c

# Fit the linear function to the subsampled data
params, params_covariance = curve_fit(linear_func, f1[1:], pavg[1:], sigma=pstd[1:])


# cut=np.random.choice(np.arange(len(f_subsampled)-15,len(f_subsampled)),8,replace=False)
# mask = np.ones(len(f_subsampled), dtype=bool)
# mask[cut] = False

# f_=f_subsampled[mask]
# pavg_=pavg_subsampled[mask]
# pstd_=pstd_subsampled[mask]


# Plot the fitted line
x_fit = np.linspace(17, 150, 100)
y_fit = linear_func(x_fit, *params)


#s=[fraster[f's{i}'] for i in range(len(fraster.files)-1)]

sns.set(style="ticks", context="paper", font_scale=1.5)

#f=fraster['f']

#sorted_indices = np.argsort(f)[::1]
#sorted_spike_times = [s[i] for i in sorted_indices]


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))  # Adjust figsize as needed

# Plot 1
for neuron_id, neuron_spike_times in enumerate(sorted_spike_times):
        ax1.vlines(neuron_spike_times, neuron_id + 0.5, neuron_id + 1.5)
        
ax1.set_yticks([1,50,100],[3,80,120])
ax1.set_xlabel('Time (ms)')
ax1.set_ylabel('Firing frequency (Hz)')

# Plot 2
ax2.errorbar(fp['avgfreqs'], fp['avgpauses'], yerr=fp['stdpauses'], fmt='o', label=r'Pause average. $\pm \ \sigma$ ', ecolor='red', capsize=2)
ax2.plot(x_fit, y_fit, label=f'Fitted line', color='blue')
ax2.set_ylabel('Pause length (ms)')
ax2.set_xlabel('Firing frequency (Hz)')
#ax2.set_xlim((15,125))
ax2.set_ylim(top=250)
ax2.legend()
ax2.grid(True)

plt.subplots_adjust(wspace=125)


# Adjust layout to make sure plots are well presented
plt.tight_layout()
