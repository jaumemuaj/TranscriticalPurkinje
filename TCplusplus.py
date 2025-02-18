"""
Created on Thu Jun 13 17:44:20 2024

@author: jaume

UPDATE 12/24:
    zlim changed from parameter to variable. It is now a function of total current
    I injected into the PC (Iint+Iexc+Iinh)
    zlim(I) parametrized according to Figure 3.16 bottom-right, for the average <z> according
    to Deltaeps_z = 8
    This update gets rid of the f/I anomaly (line 187 and 188)
"""

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

# Unit scaling for the different parameters
exp = jnp.exp
uS = 1e-6; mV = 1e-3; nA = 1e-9; nF = 1e-9; ms = 1e-3

# Input currents into Purkinje Cells
class Stim(typing.NamedTuple):
    Ipf1: float
    Ipf2: float
    Ipf3: float
    Icf : float
    Iexc: float
    Iinh: float

# Model's Parameters definition. Given a default value but can also be assigned 
# by importing a dictionary.
class Params(typing.NamedTuple):
    a: float
    b: float
    vreset: float
    wreset: float
    eps: float
    Iint: float
    C: float
    v_sp: float
    d_z: float
    el: float
    vth: float
    g_sd: float
    g_ds: float
    ad: float
    DeltaV: float
    vthd: float
    vths: float
    vd_sp: float
    Cd: float
    wdreset: float
    wiggle_vd: float
    DeltaCaV: float
    max_alpha: float
    Kcf: float
    max_vCas: float
    Kpf: float
    max_vCad: float
    Ca_half: float
    tauw: float
    tauCa: float
    tauR : float
    chi : float
    c: float
    l: float
    m: float
    g1: float
    g2: float
    g3: float
    eps_z0: float
    Deltaeps_z: float
    dres_coeff: float
    dsp_coeff : float
    dsp_coeff2: float
    slp: float
    n: float
    gl: float
    asd: float
    vsd3reset: float
    vsd2reset: float
    wsdreset: float
    vsd_sp: float
    Csd: float
    g0: float
    DeltaVsd: float
    vth3: float
    vth2: float
    sdsf3: float
    sdsf2: float
    sdsf0: float
    gamma1: float
    gamma2: float
    gamma3: float
    p: float
    w00: float
    tau_s2: float
    tau_s3: float
    zmax: float
    eta_z: float
    @classmethod
    #define the import of a dictionary to give the parameters their (scaled) values
    def init(cls, d: typing.Dict[str, float]):
        return cls(**d)
    @classmethod
    #Define default values of the parameters
    def makedefault(cls):
        default = {
            'a': 0.1*uS, 'b': -3*uS, 'vreset': -55*mV, 'wreset': 15*nA, 'eps': 1, 'Iint': 91*nA, 'C': 2*nF,
            'v_sp': -5*mV, 'd_z': 40*nA, 'el': -61*mV, 'vth':  -45*mV, 'g_sd': 3*uS, 'g_ds': 6*uS,
            'ad': 2*uS, 'DeltaV':  1*mV, 'vthd':  -40*mV, 'vths':  -42.5*mV, 'vd_sp': -35*mV, 'Cd': 2*nF, 'wdreset': 15*nA, 'wiggle_vd': 3*mV,
            'DeltaCaV': 5.16*mV, 'max_alpha':  0.09, 'Kcf': 1*nA, 'max_vCas': 80*(nA/ms), 'Kpf': 0.5*nA, 'max_vCad': 1.5*(nA/ms), 'Ca_half': 50*nA,
            'tauw': 1*ms, 'tauCa': 100*ms, 'tauR': 75*ms, 'zmax': 75*nA, 'chi':0,
            'c': 0.1*(1/mV), 'l': 0.1*(1/nA), 'm': 0.2*(1/nA), 'g1': 4*uS, 'g2': 4*uS, 'g3': 4*uS,
            'eps_z0': 10, 'Deltaeps_z': 8, 'dres_coeff': 2,'dsp_coeff':0.5, 'dsp_coeff2':0.1, 'slp': 5e-4*(1/nA), 'n': 0.2*(1/nA),
            'gl': 0.1*uS, 'asd': 0.1*uS, 'vsd3reset': -55*mV, 'vsd2reset': -50*mV, 'wsdreset': 15*nA, 'vsd_sp': -20*mV, 'Csd': 5*nF,
            'g0': 1*uS, 'DeltaVsd': 5*mV, 'vth3': -42.5*mV, 'vth2': -42.5*mV, 'sdsf3': 3.15*uS, 'sdsf2': 2.37*uS, 'sdsf0': 5,
            'gamma1': 25, 'gamma2': 25, 'gamma3': 25, 'p': 2, 'w00': 4*nA, 'tau_s2': 150*ms, 'tau_s3': 75*ms
        }
        return cls.init(default)

#Definition of state variables to track as a function of time
class State(typing.NamedTuple):
    Vs:   float | jax.Array
    Vd:   float | jax.Array
    vd1:  float | jax.Array
    vd2:  float | jax.Array
    vd3:  float | jax.Array
    sf2:  float | jax.Array
    sf3:  float | jax.Array
    w0:   float | jax.Array
    z:    float | jax.Array
    dres: float | jax.Array
    Cas:  float | jax.Array
    Cad:  float | jax.Array
    ws:   float | jax.Array
    wd:   float | jax.Array
    wd2:  float | jax.Array
    wd3:  float | jax.Array
    wd1:  float | jax.Array
    eps_z:float | jax.Array
    act:  float | jax.Array
    t:    float | jax.Array
    t_ds2:float | jax.Array
    t_ds3:float | jax.Array
    sf3:  float | jax.Array
    alpha:float | jax.Array
    zlim: float | jax.Array

    @classmethod
    #Give initial values to state variables
    def init(cls, params: Params):
        return cls(
            Vs=params.el, Vd=params.el, vd1=params.el, vd2=params.el, vd3=params.el,
            sf2=params.sdsf2, sf3=params.sdsf3, w0=params.w00, z=0, eps_z=params.eps_z0,
            dres=0, Cas=0, Cad=0, ws=0, wd=0, wd2=0, wd3=0, wd1=0, act=0, t=0.0, t_ds2=0.0, 
            t_ds3=0.0, alpha=0, zlim=0)

# Accounting of Spikes in electrical activity: Dendritic Calcium spikes at dendritic compartments
# and Action Potentials at the Soma
class Trace(typing.NamedTuple):
    state: State
    sspike: bool  | jax.Array | int
    dspike: bool  | jax.Array | int
    d3spike: bool | jax.Array | int
    d2spike: bool | jax.Array | int


#Equations to be uploaded at each timestep to keep the state cariables evolution
#PLease refer to the thesis document for understanding of all equations / variables  
# dot_x definitions correspond to derivative functions of x variable  
@partial(jax.jit, static_argnames=['dt']) # could add params to static_argnames for minor speedup
def timestep(params: Params, state: State, stim: Stim, dt: float):
    t       = state.t+dt
    
    I       = params.Iint + stim.Iinh + stim.Iexc
    Ipf     = params.gamma1*stim.Ipf1+params.gamma2*stim.Ipf2+params.gamma3*stim.Ipf3
    zeff    = state.z + state.dres
    zlim    = params.zmax + 1.02*(I-91*nA) - 1.2e-4*(I-91*nA)**2
    eps_z   = params.eps_z0 - params.Deltaeps_z/(1+exp(-params.l*(zeff-zlim)))
    #Ca-related variables
    Ca      = state.Cas + state.Cad
    act     = 1/(1+exp(-params.m*(Ca-params.Ca_half)))
    vCas    = params.max_vCas*stim.Icf / (params.Kcf+stim.Icf)
    vCad    = params.max_vCad*Ipf / (params.Kpf+Ipf)
    alpha   = (Ca<params.Ca_half)* ( params.slp*(params.Ca_half-Ca) + params.max_alpha / (1+exp(-params.n*(Ca-params.Ca_half))) ) + (Ca>params.Ca_half)* ( params.max_alpha / (1+exp(-params.n*(Ca-params.Ca_half))) )
    
    # derivatives
    dot_Vs  = (((params.el-state.Vs)**2*uS**2 + params.b*(state.Vs-params.el)*state.ws  - state.ws**2)/nA + I - zeff + params.g_sd*(state.Vd-state.Vs) )/params.C
    dot_ws  = params.eps*(params.a*(state.Vs-params.el)-state.ws+state.w0-alpha*Ca)/params.tauw
    dot_Vd  = (params.g_ds*(state.Vs-state.Vd+params.wiggle_vd)+params.g1*(state.vd3-state.Vd)+params.g1*(state.vd2-state.Vd)+params.g1*(state.vd1-state.Vd) + params.sdsf0*(params.DeltaV)*exp((state.Vd-params.vth)/(params.DeltaV))*uS-state.wd)/params.Cd
    dot_wd  = (params.ad*(state.Vd-params.el)-state.wd)/params.tauw
    dot_z   = -eps_z*state.z/params.tauCa
    dot_dres= act**params.p*params.dres_coeff*nA/ms-state.dres/params.tauR
    dot_Cas = vCas / (1+exp(-params.c*(state.Vs-params.vths))) - state.Cas/params.tauCa
    dot_Cad = vCad*(exp((state.vd3-params.vthd)/(params.DeltaCaV)) * (stim.Ipf3!=0)  + exp((state.vd2-params.vthd)/(params.DeltaCaV)) * (stim.Ipf2!=0) + exp((state.vd1-params.vthd)/(params.DeltaCaV)) * (stim.Ipf1!=0)) - state.Cad/params.tauCa
    dot_vd3 = (params.g0*(state.Vd-state.vd3)+params.gl*(params.el-state.vd3+params.wiggle_vd)+state.sf3*(params.DeltaVsd)*exp((state.vd3-params.vth3)/(params.DeltaVsd))-state.wd3+params.gamma3*stim.Ipf3)/params.Csd
    dot_wd3 = (params.asd*(state.vd3-params.el)-state.wd3)/(params.tauw)
    dot_vd2 = (params.g0*(state.Vd-state.vd2)+params.gl*(params.el-state.vd2+params.wiggle_vd)+state.sf2*(params.DeltaVsd)*exp((state.vd2-params.vth2)/(params.DeltaVsd))-state.wd2+params.gamma2*stim.Ipf2)/params.Csd
    dot_wd2 = (params.asd*(state.vd2-params.el)-state.wd2)/params.tauw
    dot_vd1 = (params.g0*(state.Vd-state.vd1)+params.gl*(params.el-state.vd1+params.wiggle_vd)-state.wd1+params.gamma1*stim.Ipf1)/params.Csd
    dot_wd1 = (params.asd*(state.vd1-params.el)-state.wd1)/params.tauw
    # events
    sspike = state.Vs > params.v_sp
    dspike = state.Vd > params.vd_sp
    d3spike = state.vd3 > params.vsd_sp
    d2spike = state.vd2 > params.vsd_sp
    # updates
    Vs = jax.lax.select(sspike, params.vreset, state.Vs + dot_Vs*dt)
    ws = jax.lax.select(sspike, params.wreset, state.ws + dot_ws*dt)
    Vd = jax.lax.select(dspike, params.vreset, state.Vd + dot_Vd*dt)
    wd = state.wd + dot_wd*dt
    z = state.z + dot_z*dt
    dres = state.dres + dot_dres*dt
    Cas = state.Cas + dot_Cas*dt
    Cad = state.Cad + dot_Cad*dt
    vd3 = jax.lax.select(d3spike, params.vsd3reset, state.vd3 + dot_vd3*dt)
    wd3 = state.wd3 + dot_wd3*dt
    vd2 = jax.lax.select(d2spike, params.vsd2reset, state.vd2 + dot_vd2*dt)
    wd2 = state.wd2 + dot_wd2*dt
    vd1 = state.vd1 + dot_vd1*dt
    wd1 = state.wd1 + dot_wd1*dt
    z = z \
          + sspike * (params.d_z) \
          + d3spike * (params.dsp_coeff*params.d_z) \
          + d2spike * (params.dsp_coeff*params.d_z)
    dres= dres \
            + d3spike * (params.dsp_coeff2*params.d_z) \
            + d2spike * (params.dsp_coeff2*params.d_z)
    wd = jax.lax.select(dspike, wd + params.wdreset, wd)
    wd3 = jax.lax.select(d3spike, wd3 + params.wsdreset, wd3)
    wd2 = jax.lax.select(d2spike, wd2 + params.wsdreset, wd2)
    # ignored?
    t_ds3=state.t_ds3
    t_ds3=jax.lax.select(d3spike, state.t, state.t_ds3)
    t_ds2=state.t_ds2
    t_ds2=jax.lax.select(d2spike, state.t, state.t_ds2)

    sf2 = params.sdsf2*(1-0.9*exp(-(state.t-state.t_ds2)/params.tau_s2))
    sf3 = params.sdsf3*(1-0.9*exp(-(state.t-state.t_ds3)/params.tau_s3))
    w0 = state.w0
    state_next = State(Vs=Vs, Vd=Vd, vd1=vd1, vd2=vd2, vd3=vd3, sf2=sf2, sf3=sf3, w0=w0, z=z, dres=dres, alpha=alpha,
                        Cas=Cas, Cad=Cad, ws=ws, wd=wd, wd2=wd2, wd3=wd3, wd1=wd1, eps_z=eps_z, act=act, t=t, t_ds3=t_ds3, t_ds2=t_ds2,
                        zlim=zlim)
    trace = Trace(state=state, sspike=sspike, dspike=dspike, d3spike=d3spike, d2spike=d2spike)
    return state_next, trace

#FUNCTION TO SIMULATE A TRACE UNDER POSSIBLE PF OR CF INPUT
def simulation(dict_params,PF=(0,0,0),CF=0,isi=0,plots=True):
    """
    

    Parameters
    ----------
    dict_params : DICTIONARY
        DICTIONARY OF PARAMETERS TO INPUT INTO THE MODEL.
    PF : THRUPLE, optional
        FIRST NUMBER IS THE DURATION OF THE PF INPUT, SECOND ONE IS THE TOTAL 
        CURRENT AMPLITUDE FROM A PF BUNDLE, THIRD NUMBER IS THE DENDRITIC TREE
        IT IS TARGETTING. The default is (0,0,0).
    CF : FLOAT, optional
        INTENSITY, IN nA, OF THE CF INPUT. 
        The default is 0, and for the standard CF input a value of 10nA is used.
    isi : float, optional
        delay in CF input from default time. The default is 0.
    plots : bool, optional
        DO YOU WANT THE PLOTS OF THE TRACE TO BE OUTPUTTED (TRUE) OR JUST THE 
        DICTIONARY WITH ALL THE DATA (FALSE)? IF TRUE, OUTPUTS PLOTS FOLLOWING DIFFERENT
        VARIABLES. The default is True.

    Returns
    -------
    traces : DICTIONARY
        CONTAINS THE ARRAYS WITH THE TEMPORAL EVOLUTION OF ALL RELEVANT VARIABLES.

    """
    nsteps = 50000 # 50000 steps = 1 second
    dt = 0.02*ms
    t = dt * jnp.arange(nsteps)
    
    
    # Generate the Iexc current with the specified pattern
    # This lines generate a increasing step current but are not applied as long as
    # there is a 0 multiplying the stims definition
    Iexc = jnp.zeros(nsteps)
    Iinj = 80 * nA
    for i in range(0, len(Iexc), int(250*ms / dt)):
        if i + int(200*ms / dt) < len(Iexc):
            Iexc = Iexc.at[int(i + 100*ms / dt):int(i + 200*ms / dt)].set(Iinj)
            Iinj += 10 * nA
    # Create the stimulus with the new Iexc pattern
    
    #Time at which CF input arrives. Set to a default + any delay (isi)
    tcf=500+isi
    #Time at which PF input arrives.
    tpf=500
    

    dur,iamp,sd=PF
    
    #Assign the corresponding PF inout ro the dendritic tree it pertains
    pfs=[0,0,0]
    #Assign the current to the targeted dendritic tree (1,2 or 3)
    pfs[sd-1]=iamp
    i1,i2,i3=pfs
    
    durs=[0,0,0]
    durs[sd-1]=dur
    dur1,dur2,dur3=durs
    
    #Assignment of both PF and CF stimulus. Somehow the stimulus loop every second of
    #simulation. Not sure why at the moment.
    stims = Stim(
        Ipf1=i1 * nA * ((t % 1 > (tpf) * ms) & (t % 1 < (tpf+dur1) * ms)),
        Ipf2=i2 * nA * ((t % 1 > (tpf) * ms) & (t % 1 < (tpf+dur2) * ms)),
        Ipf3=i3 * nA * ((t % 1 > (tpf) * ms) & (t % 1 < (tpf+dur3) * ms)),
        Icf=CF * nA * ((t % 1 > (tcf) * ms) & (t % 1 < (tcf+5) * ms)),
        #Icf=jnp.zeros(nsteps),
        Iexc=0*Iexc,
        Iinh=0*t
    )   
   
    #Initiate parameter values and variables
    params = Params.init(dict_params)
    state0 = State.init(params)
    
    
    a = time.time()
    _, trace = jax.lax.scan(
        lambda state, stim: timestep(params, state, stim, dt),
        state0, stims
    )
    
    b = time.time()
    #print('simulation took:', b - a, 'seconds')
        
    
    
    #Traces to be saved and outputted
    vs = trace.state.Vs.at[trace.sspike].set(params.v_sp) #all 
    vd= trace.state.Vd.at[trace.dspike].set(params.vd_sp)
    vd1=trace.state.vd1
    vd2=trace.state.vd2
    vd3=trace.state.vd3
        
    
    dres = trace.state.dres
    ca= trace.state.Cas + trace.state.Cad
    z= trace.state.z
    eps_z=trace.state.eps_z
    act=trace.state.act
    ws=trace.state.ws
    alpha=trace.state.alpha
    
    zlim=trace.state.zlim
    
    timestamp=516
    # print(eps_z[int(timestamp/0.02)])
    # print(dres[int(timestamp/0.02)]/nA)
    # print(z[int(timestamp/0.02)]/nA)
    # print(act[int(timestamp/0.02)])

    # Plotting
    spike_indices = jnp.where(trace.sspike)[0]
    spike_times = spike_indices * dt *1e3
    
    #Instantaneous frequencies at each spike
    freq=jnp.round(jnp.array([1/(spike_times[i+1]-spike_times[i]) for i in range(len(spike_times)-1)])*1000,2)
    #inter-spike interval aka pauses in firing between each spike
    pauses=jnp.round(jnp.array([(spike_times[i+1]-spike_times[i]) for i in range(len(spike_times)-1)]),2)
    
    if CF!=0:
        #Center the time at the time of the CF
        tcs=t-tcf*1e-3
    else:
        tcs=t
    if plots==True:
        _, ax = plt.subplots(figsize=(14,10),nrows=6, sharex=True, gridspec_kw=dict(hspace=0.1), height_ratios=[1,1,1,1,1,1])
        # Plot one voltage trace on the top plot
        ax[0].plot(tcs*1e3,ca/nA,label='ca')
        ax[0].legend()
        
        ax[1].plot(tcs * 1e3, act,label=r'$g_{KCa}$')
        ax[1].legend()
        #ax[0].set_xlim((450,800))
        ax[2].plot(tcs*1e3, dres/nA,label='KCa')
        ax[2].plot(tcs*1e3, z/nA,label='KV')
        ax[2].plot(tcs*1e3, (z+dres)/nA)
        ax[2].legend()
    
        # Plot the Iexc current trace on the bottom plot
        ax[3].plot(tcs * 1e3, eps_z, label="Recovery rate", color='r')
        ax[3].legend()
        
        ax[4].plot(tcs*1e3,vs/mV,label='Soma')
        ax[4].plot(tcs*1e3,vd/mV,label='M0')
        ax[4].plot(tcs*1e3,vd1/mV,label='S1 ')
        ax[4].plot(tcs*1e3,vd2/mV,label='S2')
        ax[4].plot(tcs*1e3,vd3/mV,label='S3')


        ax[4].legend()
        #ax[4].set_xlim((-100,300))
        ax[4].set_xlabel('Time (ms)')
        
        _,_,tfreq,freq=smooth_freq([spike_times],bin_size=10, start_time=0, end_time=int(nsteps*dt/ms), sigma=20,plot=False)
        
        if CF!=0:
            ax[5].plot(tfreq-tcf,freq[0],label='frequency')
        else:
            ax[5].plot(tfreq,freq[0],label='frequency')
        ax[5].legend()
        
        plt.tight_layout()
        plt.show()
            
    
    traces={}
    traces['Soma']=vs;traces['M0']=vd;traces['S1']=vd1;traces['S2']=vd2;traces['S3']=vd3
    traces['t']=t;traces['spikes']=spike_times
    traces['Ca']=ca;traces['z']=z;traces['dres']=dres;traces['eps_z']=eps_z;traces['act']=act
    traces['ws']=ws; traces['alpha']=alpha;
    traces['pauses']=pauses; traces['freq']=freq; traces['zlim']=zlim
    
    return traces



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
   
def plot_sorted_raster(spike_times, neuron_values, title='Sorted Raster Plot',yticks=False):
    """
    Plot a raster plot of different trials of a PC

    Parameters
    ----------
    spike_times : list of lists
        contains arrays of spike times from the different trials.
    neuron_values : list
        factor used to classify the raster by. example: maximum pause --> raster
        plot is displayed with increasing maximum pause length.
    title : string, optional
        title. The default is 'Sorted Raster Plot'.

    Returns
    -------
    the plot.

    """
    # Sort the neurons based on neuron_values in descending order
    sorted_indices = np.argsort(neuron_values)[::1]
    # Create a new sorted spike_times array
    sorted_spike_times = [spike_times[i] for i in sorted_indices]
    plt.figure(figsize=(10, 6))
    
    # Iterate through each neuron and its spike times
    for neuron_id, neuron_spike_times in enumerate(sorted_spike_times):
        plt.vlines(neuron_spike_times, neuron_id + 0.5, neuron_id + 1.5)
    
    
    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron')
    plt.title(title)
    #plt.ylim(0.5, len(spike_times) + 0.5)
    if yticks==True:
        plt.yticks(np.linspace(1, len(spike_times) + 1,3), 
                   np.round(np.array([min(neuron_values),round(np.mean(neuron_values),2),max(neuron_values)]),1))
    
    
    