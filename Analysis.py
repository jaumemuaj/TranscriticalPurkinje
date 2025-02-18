#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 21:49:25 2024

@author: jaume
"""

from TCplusplus import *
import seaborn as sns
import numpy as np

#A simple use of the simulation function, which gives you a time evolution of
#a PC as described by the TC+ model

sns.set(style="ticks", context="paper", font_scale=1.5)

with open('params1606.json', 'r') as file:
    params=json.load(file)

# Example_trace=simulation(parameter_dictionary,PF input, CF input, delay of CF
#                          (default at 0), do you want the plots?)

tr=simulation(params,PF=(0,0.2,2),CF=10,plots=True)

#print(tr['zlim'])
#%%

#MAIN 2 gives population variability, batches of 1
# Creates a population of N PCs, by giving variability to the parameters in
# VARIATION list of a PCT percentage
#similar to the simulation function, can for sure be improved
def main2():
    nsteps = 50000 # 50000 steps = 1 second
    dt = 0.02*ms
    t = dt * jnp.arange(nsteps)
    N=200
    pct=0.05   
    variation=['Iint','zmax','Deltaeps_z','max_vCas','dres_coeff','m','tauR','Ca_half','slp','tauCa','C',
          'Cd','Csd','max_vCad','Kpf','Kcf','vths','c','p']
    
    #Spike times
    spike_raster=[]
    #List of pauses between spikes
    pauselist=[]
    
    #will take the average frequency of all the population and its standard deviation.
    avg_freqs=[]
    std_freqs=[]
    
    #This loop could be optimized
    Is=[0]
    for idx,Iinj in enumerate(Is):
        for i in range(N):
            with open('params1606.json', 'r') as file:
                params=json.load(file)
            params['Iint']=92e-9
            #params['Deltaeps_z']=4
            #params['dres_coeff']=2.5
            
            #Give each trial new parameters according to the variability
            var=pct*(np.round(np.random.rand(len(variation)),3)*2-1)
            new_params={}
            for param,v in zip(variation,var):
                params[param]+=v*params[param]
                new_params[param]=params[param]
            
            #These are 'junk' lines, twould determine changes in the input current 
            calibration=200*ms   
            Iexc = np.zeros(nsteps)
            IinjnA = Iinj * nA
            Iexc[int(calibration/dt):]=IinjnA
            Iexc=jnp.array(Iexc)
            
                    
            tcf=500
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
        
    #returns spike times for every trial, with the list of pauses and average frequency
    #and standard deviation
    return spike_raster,pauselist,Is,avg_freqs,std_freqs

#s,p,Is,f,std=main2()

#%%

#MAIN3 creates batches of N trials at different total current (Iint + Iinj)
#which can later be used to create f/I plots with current injection in soma

#Check Stims for CF current, tcf and tpd for cf and pf input placement
def main3():
    nsteps = 50000 # 50000 steps = 1 second
    dt = 0.02*ms
    t = dt * jnp.arange(nsteps)
    N=10
    pct=0.05
    variation=['Iint','zmax','Deltaeps_z','max_vCas','dres_coeff','m','tauR','Ca_half','slp','tauCa','C',
          'Cd','Csd','max_vCad','Kpf','Kcf','vths','c','p','d_z']
    
    
    spike_raster=[]
    pauselist=[]
    avg_freqs=[]
    std_freqs=[]
    
    #make average of each batch (same injected current) and its standard deviation
    avg_pauses=[]
    std_pauses=[]
    
    #these track the values of eps_z and z in order to make figure 3.16 of the thesis
    epszs=[]
    zs=[]
    steps=30
    Is=np.linspace(-10,350,30)
    
    trial=0
    for idx,Iinj in enumerate(Is):
        for i in range(N):
            with open('params1606.json', 'r') as file:
                params=json.load(file)
                params['Iint']=92e-9+Iinj*1e-9
                params['Deltaeps_z']=deltaeps
                
            if i==0:
                #print(params['Deltaeps_z'])
                tr=simulation(params,plots=False)
                zs.append(np.mean(tr['z'][int(300/0.02):int(800/0.02)]))
                epszs.append(np.mean(tr['eps_z'][int(300/0.02):int(800/0.02)]))
            
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
                Icf=0 * nA * ((t % 1 > tcf * ms) & (t % 1 < (tcf+5) * ms)),
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
        
    return output, epszs,zs

freqs=[]
eps=[]
z=[]
stds=[]

#loop over different values of Deltaeps_z (part of making figure 3.16)
for deltaeps in [8]:
    info,epszs,zs=main3()
    s=info['spikeraster'];p=info['pauselist'];Is=info['Iinj'];f=info['avgfreqs'];std=info['stdfreqs']
    pavg=info['avgpauses'];pstd=info['stdpauses']
    freqs.append(f)
    stds.append(std)
    eps.append(epszs)
    z.append(zs)
#%%

#Make f/I plot
#s is defined as the spikeraster  of a batch (only one for main2)
cap=len(s)
plt.figure()
plt.errorbar(Is, f, yerr=std, fmt='o', label='Data with error bars', capsize=5)
plt.ylabel('Frequency (Hz)')
plt.xlabel('Iinj (needs scaling)')

#%%

zss=[float(z) for z in zs]

plt.plot(Is,zss)

#%%

# Example data


# Fit a quadratic equation (degree=2)
coefficients = np.polyfit(x, y, deg=2)

# Extract coefficients
a, b, c = coefficients
print(f"Fitted equation: y = {a:.2f} + {b:.2f}x + {c:.2f}x^2")

# Generate fitted y values for visualization
x_fit = np.linspace(min(x), max(x), 100)  # Generate smooth x values
y_fit = a + b * x_fit + c * x_fit**2

# Plot original data and fitted curve
plt.scatter(x, y, color='blue', label='Original Data')  # Data points
plt.plot(x_fit, y_fit, color='red', label='Fitted Curve')  # Fitted quadratic line
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Quadratic Fit')
plt.show()
#%%
from scipy.optimize import curve_fit
x = Is
y = zss
def quadratic(x, a, b, c):
    return a + b * x + c * x**2

# Fit the quadratic function to the data
popt, pcov = curve_fit(quadratic, x, y)

# Extract coefficients
a, b, c = popt
print(f"Fitted equation: y = {a:.2f} + {b:.2f}x + {c:.2f}x^2")

# Generate fitted y values for visualization
x_fit = np.linspace(min(x), max(x), 100)  # Generate smooth x values
y_fit = quadratic(x_fit, *popt)  # Use the fitted parameters to calculate y values

# Plot original data and fitted curve
plt.scatter(x, y, color='blue', label='Original Data')  # Data points
plt.plot(x_fit, y_fit, color='red', label='Fitted Curve')  # Fitted quadratic curve
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Quadratic Fit Using curve_fit')
plt.show()
#%%
#RASTER plot (after execution of main2 function)

fig, ax = plt.subplots(figsize=(8, 6))
sns.set(style="ticks", context="paper", font_scale=1.5)

# Plot the sorted raster plot
plot_sorted_raster(ax,s, p,yticks=False)
#plt.yticks([1,50,100],[3,80,120])
#plt.xlim((200,1000))
ax.set_ylabel('Neuron nº')
plt.show()
print(np.mean(p))



