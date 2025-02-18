#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 17:06:10 2025

@author: jaume
"""

from TCplusplus import *


import json
from cmaes import CMA
from scipy.signal import convolve
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import itertools
import time
from functools import partial
import typing
import jax
import jax.numpy as jnp
import seaborn as sns
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import griddata
import matplotlib.gridspec as gridspec



#timestep (seconds)
dt=0.02

#Interval analyzed for Van Rossum distance score
start,stop=[450,800]

#Van Rossun decay constant
tau=25


#Exponential kernel function
def exponential_kernel(tau, length, dt):
    t = np.arange(0, length, dt)
    return (1/tau) * np.exp(-t/tau)


#Input an array of spiking times and return a binarized sequence of those same spikes
#in the selected interval
def binarize_spike_train(spikes, interval, dt):
    start,stop=interval
    bins = np.arange(start, stop + dt, dt)
    binned_spikes = np.zeros_like(bins)
    indices = np.searchsorted(bins, spikes)
    indices = indices[indices < len(binned_spikes)]
    binned_spikes[indices] = 1
    return binned_spikes[:-1]  # Exclude last bin to match time vector length


#turn the binarized spike traines into a sequence of exponential kernels and compute
#van Rossum distance
def vanrossum_distance(train1, train2, interval, tau, dt=0.02):
    start,stop=interval
    kernel = exponential_kernel(tau, 5 * tau, dt)  # Kernel length is 5*tau
    spike_train1 = binarize_spike_train(train1, interval, dt)
    spike_train2 = binarize_spike_train(train2, interval, dt)
    continuous1 = convolve(spike_train1, kernel, mode='full')[:len(spike_train1)]
    continuous2 = convolve(spike_train2, kernel, mode='full')[:len(spike_train2)]
    distance = np.sqrt(np.sum((continuous1 - continuous2) ** 2)) * ((stop-start)*1e-3)
    return distance

#Generate ararys of traces of a PC on all 5 compartments from previously saved .npz files
#(from ZdS model simulations with specific PF stimulus duration, intensity and dendritic target; or CF stimulus)
#Outputs both electronic activity of all compartments and a list of spiking times in the soma
def open_trace_data(dur_stim,iamp,dend,CF=False,isi=0):
    """
    Parameters
    ----------
    dur_stim, iamp, dend are arrays but so far (13/06) only with 1 element
    CF : TYPE, optional
        DESCRIPTION. The default is False.
    isi  is a number

    """
    compartments=['Soma','M0','S1','S2','S3']
    data_traces={}
    if CF==False:
        file_prefix = f'singlePF/{dur_stim[0]}-{int(1000 * iamp[0])}-{dend[0]}'
        with np.load(f'{file_prefix}/traces_{dur_stim[0]}{int(1000*iamp[0])}{dend[0]}.npz') as data:
            data_traces['t']=data['t']
            for comp in compartments:
                data_traces[comp]=data[comp]
        with np.load(f'{file_prefix}/spikes_{dur_stim[0]}{int(1000*iamp[0])}{dend[0]}.npz') as spikes:
            data_traces['spikes']=spikes['arr_0']
    elif CF==True:
        with np.load(f'singleCF/cfdelay{isi}-0-0-0/traces.npz') as data:
            data_traces['t']=data['t']
            for comp in compartments:
                data_traces[comp]=data[comp]
            data_traces['S4']=data['S4']
        with np.load(f'singleCF/cfdelay{isi}-0-0-0/spike_times.npz') as spikes:
            data_traces['spikes']=spikes['arr_0']
    return data_traces


#Return the van Rossum score from comparing two spike trans (dspikes --> ZdS model, mspikes --> TC+ model)
def score_no(_id_,dtraces,mtraces,starttime,stoptime):
  
    if _id_=='vanrossum':
        dspikes=dtraces['spikes']
        mspikes=mtraces['spikes']
        dspikes=dspikes[(dspikes>starttime) & (dspikes<stoptime)]
        mspikes=mspikes[(mspikes>starttime) & (mspikes<stoptime)]
        
        score=vanrossum_distance(dspikes, mspikes, [starttime,stoptime], tau)
    
        return score
    else:
        raise NotImplementedError(f"Method {_id_} not implemented.")
        
        
     
#Calculates the VR score between ZdS and TC+ models for a specific experiment = (dur_stim, current input, target dendrite)
def evaluate(params, experiment, context, method='vanrossum',isi=0,tau=tau,starttime=start,stoptime=stop):
    """
    Parameters
    ----------
    params : TYPE
        DESCRIPTION.
    run  : list or tuple
    
    starttime : TYPE, optional
        DESCRIPTION. The default is start.
    stoptime : TYPE, optional
        DESCRIPTION. The default is stop.
    method : TYPE, optional
        DESCRIPTION. The default is 'vanrossum'.

    Returns
    -------
    total_score : TYPE
        DESCRIPTION.

    """
    dur_stim,iamp,dend=experiment
    
    dt=0.02
    
    total_score=0
    
    if context=='singlePF':
        #for d in dur_stim: for extending to multiple
        if dend==1:
                idx_dendrites=4
        else:
            idx_dendrites=dend
        
        dtraces=open_trace_data([dur_stim], [iamp], [idx_dendrites])
        mtraces = simulation(params,PF=(dur_stim,iamp,dend),plots=False)
        
        total_score+=score_no(method,dtraces,mtraces,starttime,stoptime)
    
    elif context=='singleCF':
        dtraces=open_trace_data([0],[0],[0], CF=True,isi=isi)
        mtraces = simulation(params,CF=10,isi=isi,plots=False)
        total_score+=score_no(method,dtraces,mtraces,starttime,stoptime)
    
    return total_score


#Total VR score when doing multiple runs at once (same duration, same input, all 3 dendritic compartments)
#only for PF stimulus
def vr_scores(params,runs,tau=10):
    

    runs1=[]
    for run in runs:
        for d in [1,2,3]:
            runs1.append(run+(d,)) #append each of the dendritic compartments where PF input is targeted

    runscores=[]
    for run in runs1:
        score=0
        score+=evaluate(params,run,'singlePF',tau=tau)
        runscores.append(score)
    
    return runscores, runs1


def colormap_coords(entries, scores):
    """
    Parameters:
    - entries: list of tuples, each containing (x, y, z)
    - scores: list of scores corresponding to each entry
    - title: title of the plot
    """
    # Sum scores in groups of three
    summed_scores = [sum(scores[i:i+3]) for i in range(0, len(scores), 3)]

    # Extract x, y coordinates from entries in groups of three
    coords = [(entries[i][0], entries[i][1]*1000) for i in range(0, len(entries), 3)]

    # Determine the size of the grid
    x_coords, y_coords = zip(*coords)
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    
    print(x_coords)
    print(y_coords)
    print(summed_scores)
    print(np.mean(summed_scores))

    return x_coords,y_coords,summed_scores



def colormap_plot(ax,x_coords, y_coords, summed_scores, title='Van-Rossun Distance Score (Normalized)', method='cubic'):
    """
    Plot a 2D colormap with the color showing the magnitude of summed scores.

    Parameters:
    - x_coords: list of x coordinates
    - y_coords: list of y coordinates
    - summed_scores: list of summed scores
    - title: title of the plot
    - method: method of interpolation ('linear', 'nearest', 'cubic')
    """
    
    summed_scores = np.array(summed_scores)
    best_score=2
    worst_score=4.5
    normalized_scores = 1 - (summed_scores - best_score) / (worst_score - best_score)
    
    # Invert the scores if lower is better
    #inverted_scores = 1 - normalized_scores
    # Create grid values first.
    xi = np.linspace(min(x_coords), max(x_coords), 300)
    yi = np.linspace(min(y_coords), max(y_coords), 300)
    xi, yi = np.meshgrid(xi, yi)

    # Interpolate the scores
    zi = griddata((x_coords, y_coords), normalized_scores, (xi, yi), method=method)

    # Plot the 2D colormap
    im=ax.imshow(zi, extent=(min(x_coords), max(x_coords), min(y_coords), max(y_coords)),
               origin='lower', cmap='RdBu', aspect='auto',vmin=0.5, vmax=1)
    fig.colorbar(im,label='Normalized VR score')
    #plt.scatter(x_coords, y_coords, c='darkred', edgecolor='k')  # Optional: highlight original points
    ax.set_xlabel('Duration of PF stimulus (ms)')
    ax.set_ylabel('Current amplitude (pA)')
    ax.set_xticks([15,25,30,50,75,100]) #these are the tested PF input durations (check zds_runs)
    ax.set_yticks([50,100,150,200]) #input currents tested in pA (check zds_runs)
    #ax.set_title(title)
    
   
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
    firing_rates = np.array(firing_rates) #all the f traces

    # Calculate the mean firing rate across all neurons
    mean_freq_trace = np.mean(firing_rates, axis=0)
    
    
    id_freq_avg=np.array([np.mean(f) for f in firing_rates]) #avg f rate for each neuron
    
    
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
    

with open('params1606.json', 'r') as file:
    params=json.load(file)

#Experimental trials from the ZdS model that are going to be analyzed in the figure
#imported from presavez .npz files

zds_runs=[(15,0.15),(15,0.2),
          (25,0.1),(25,0.15),
          (30,0.1),(30,0.15),(30,0.2),
          (50,0.05),(50,0.1),(50,0.15),
          (75,0.05),(75,0.1),
          (100,0.05),(100,0.1)
          ]

scores2,runs2=vr_scores(params, zds_runs)
x,y,s=colormap_coords(runs2, scores2)

sns.set(style="ticks", context="paper", font_scale=1.5)

fig = plt.figure(figsize=(12, 12))
gs = gridspec.GridSpec(5, 2, height_ratios=[4, 1, 1, 1, 1])  # Added a small height for the ghost axis
gs.update(hspace=0)

# Color map plot
ax1 = plt.subplot(gs[0, :])
colormap_plot(ax1,x, y, s)


#params['sdsf3']=2.75e-6
tr1=open_trace_data([50], [0.15], [2])
tr2=open_trace_data([50], [0.1], [3])
tr11=simulation(params,PF=(50,0.15,2),plots=False)
tr22=simulation(params,PF=(50,0.1,3),plots=False)


# Ghost axis to ensure zero hspace between rows
ghost_ax = plt.subplot(gs[1, :])
ghost_ax.axis('off')  # Hide the ghost axis

# Trace comparison plots
for col, (trace, trace_comp, title) in enumerate([
    (tr1, tr11, '50 - 150 - S2'),
    (tr2, tr22, '50 - 100 - S3'),
]):
    ax_top = plt.subplot(gs[2, col])
    ax_bottom = plt.subplot(gs[3, col], sharex=ax_top)
    
    ax_top.plot(trace['t'], trace['Soma'], label='Original Trace')
    ax_top.plot(trace['t'], trace['S2'], label='Original Trace', color='darkmagenta')
    ax_top.plot(trace['t'], trace['S3'], label='Original Trace', color='olive')

    ax_top.set_ylabel('mV')
    ax_top.axis('off')
    #ax_top.set_title(title)

    #ax_top.legend()
    ax_top.label_outer()  # Only show outer labels to avoid overlap

    ax_bottom.plot(trace_comp['t']/ms, trace_comp['Soma']/mV,color='tomato')
    ax_bottom.plot(trace_comp['t']/ms, trace_comp['S2']/mV,color='darkmagenta')
    ax_bottom.plot(trace_comp['t']/ms, trace_comp['S3']/mV, label='Comparison 1',color='olive')

    ax_bottom.set_ylabel('mV')
    ax_bottom.axis('off')


    ax_bottom.set_xlim((400,800))
    ax_bottom.set_xlabel('Time')
    #ax_bottom.legend()
    ax_bottom.label_outer()  # Only show outer labels to avoid overlap

ax_top.set_xticklabels(np.arange(-100,350,50))

_,_,t,fr=smooth_freq([tr1['spikes'],tr11['spikes']], bin_size=10, start_time=200, end_time=1000, sigma=30,plot=False)

ax6=plt.subplot(gs[4,0], sharex=ax_top)

ax6.plot(t,fr[0],label='ZdS')
ax6.plot(t,fr[1],label='TC+',color='tomato')
#ax6.set_ylabel('Hz')
#ax6.legend(loc='upper right',fontsize=10)
ax6.spines['top'].set_visible(False)
ax6.spines['right'].set_visible(False)
ax6.spines['left'].set_visible(False)
ax6.yaxis.set_ticks_position('none')
ax6.tick_params(axis='y', which='both', left=False, labelleft=False)
ax6.spines['bottom'].set_position(('outward', 10))  # You can adjust the outward position if needed
ax6.xaxis.set_ticks_position('bottom')
ax6.xaxis.set_label_position('bottom')
ax6.set_xlabel('ms')
_,_,t,fr=smooth_freq([tr2['spikes'],tr22['spikes']], bin_size=10, start_time=200, end_time=1000, sigma=30,plot=False)

ax7=plt.subplot(gs[4,1], sharex=ax_top,sharey=ax6)

ax7.plot(t,fr[0],label='ZdS')
ax7.plot(t,fr[1],label='TC+',color='tomato')
#ax7.set_ylabel('Hz')
ax7.spines['top'].set_visible(False)
ax7.spines['right'].set_visible(False)
ax7.spines['left'].set_visible(False)
ax7.yaxis.set_ticks_position('none')
ax7.tick_params(axis='y', which='both', left=False, labelleft=False)
ax7.spines['bottom'].set_position(('outward', 10))  # You can adjust the outward position if needed
ax7.xaxis.set_ticks_position('bottom')
ax7.xaxis.set_label_position('bottom')
ax7.set_xlabel('ms')
ax7.legend(loc='upper right',fontsize=10)
ax7.label_outer()



plt.tight_layout()

#plt.savefig('colormap1.png',dpi=500)
plt.show()
