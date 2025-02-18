#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 17:54:53 2024

@author: jaume
"""

from TCplusplus import *
import numpy as np
import itertools

with open('params1606.json', 'r') as file:
    params=json.load(file)
    
# parameter_ranges = {
#     'tauR': [5e-3,125e-3],   # Range for the first dimension
#     'dres_coeff': [0.5, 4],   # Range for the second dimension
#     'Iint': [85e-9, 120e-9],   # Range for the third dimension
#     'vths': [-50e-3, -40e-3],   # Range for the fourth dimension
#     'max_vCas': [50e-6, 120e-6],   # Range for the fifth dimension
#     'Ca_half': [25e-9, 75e-9],   # Range for the sixth dimension
#     #'g_sd': [1e-6, 5e-6],   # Range for the seventh dimension
#     #'g_ds': [2e-6, 10e-6]    # Range for the eighth dimension
# }

#For firing frequency

parameter_ranges = {
    #'tauR': [5e-3,150e-3],   # Range for the first dimension
    #'dres_coeff': [0.5, 4],   # Range for the second dimension
    'Iint': [80e-9, 120e-9],   # Range for the third dimension
    #'vths': [-42.5e-3, -42.6e-3],   # Range for the fourth dimension
    'eps_z0': [5, 20],   # Range for the fifth dimension
    'g1': [0.5e-6, 10e-6],   # Range for the sixth dimension
    'g_sd': [0.5e-6, 10e-6],   # Range for the seventh dimension
    'g_ds': [0.5e-6, 10e-6],    # Range for the eighth dimension
    'g0':[0.5e-6, 10e-6]
}


param_names = list(parameter_ranges.keys())
print(param_names)


# Define the number of steps for each dimension
num_steps = 4  # For simplicity, use 2 steps. Adjust as needed.

# Create the 8D matrix
dims = [num_steps] * len(param_names)
A = np.zeros(dims)

# Generate all possible values for each parameter
param_values = {param: np.linspace(start, stop, num_steps) for param, (start, stop) in parameter_ranges.items()}
print(param_values)

def mean_cf_pause(params,N):
    pause_list=[]
    pct=0.05
    variation=['Iint','zmax','Deltaeps_z','max_vCas','dres_coeff','m','tauR','Ca_half','slp','tauCa','C',
          'Cd','Csd','max_vCad','Kpf','Kcf','vths','c','p',]
    for n in range(N):
        run_params=params.copy()
        var=pct*(np.round(np.random.rand(len(variation)),3)*2-1)
        for param,v in zip(variation,var):
            run_params[param]+=v*run_params[param]
            
        tr=simulation(run_params,CF=10,plots=False)
        pause_list.append(np.max(tr['pauses']))
    
    return np.mean(pause_list)

def mean_avg_freq(params,N):
    freq_list=[]
    pct=0.05
    variation=['Iint','zmax','Deltaeps_z','max_vCas','dres_coeff','m','tauR','Ca_half','slp','tauCa','C',
          'Cd','Csd','max_vCad','Kpf','Kcf','vths','c','p',]
    for n in range(N):
        run_params=params.copy()
        var=pct*(np.round(np.random.rand(len(variation)),3)*2-1)
        for param,v in zip(variation,var):
            run_params[param]+=v*run_params[param]
            
        tr=simulation(run_params,plots=False)
       # print(len(tr['freq']))
        #if len(tr['freq'])==0:
            #print('Spikes', tr['spikes'])
        freq=tr['freq'][10:-10]
        #print(freq)
        #print(freq)
        freq_list.append(np.mean(freq))
    
    #print(np.mean(freq_list))
    
    return np.mean(freq_list)



all_indices = list(itertools.product(*[range(d) for d in dims]))
#%%
# PAUSE!!!!
count=0
completion=0
for indices in all_indices:
    if count % (len(all_indices)//100) == 0:
        print(f'Progress: {completion}% ; {count}/{len(all_indices)} done')
        completion+=1
        
    with open('params1606.json', 'r') as file:
        base_params=json.load(file)
    
    for i, param in enumerate(param_names):
        base_params[param] = param_values[param][indices[i]]
        
    A[indices] = mean_cf_pause(base_params, 10)
    count+=1
#%%
# F!!!!
count=0
completion=0
for indices in all_indices:
    if count % (len(all_indices)//100) == 0:
        print(f'Progress: {completion}% ; {count}/{len(all_indices)} done')
        completion+=1
        
    with open('params1606.json', 'r') as file:
        base_params=json.load(file)
    
    for i, param in enumerate(param_names):
        base_params[param] = param_values[param][indices[i]]
        #print(param, base_params[param])
        
    A[indices] = mean_avg_freq(base_params,1)
    count+=1
#%%


def plotMultiD(A, param_names, row_params, col_params):
    dims = A.shape  # Get the dimensionality of A

    # Map param names to their indices
    param_indices = {name: idx for idx, name in enumerate(param_names)}

    # Get the indices for the row and column parameters
    row_indices = [param_indices[param] for param in row_params]
    col_indices = [param_indices[param] for param in col_params]

    # Calculate total number of rows and columns
    num_rows = np.prod([dims[i] for i in row_indices])
    num_cols = np.prod([dims[i] for i in col_indices])

    # Initialize the 2D matrix
    N = np.zeros((num_rows, num_cols))

    # Generate all combinations of indices for the given dimensions
    all_indices = list(itertools.product(*[range(d) for d in dims]))

    # Construct row and column index mappings
    def calculate_index(indices, order):
        index = 0
        for i, dim in enumerate(order):
            stride = np.prod([dims[order[j]] for j in range(i + 1, len(order))]) if i + 1 < len(order) else 1
            index += indices[dim] * stride
        return index

    # Evaluate the function for each combination of indices and populate the matrix
    for indices in all_indices:
        row_idx = calculate_index(indices, row_indices)
        col_idx = calculate_index(indices, col_indices)
        N[row_idx, col_idx] = A[indices]
    viridis = plt.cm.get_cmap('viridis', 256)  # Get 256 colors from viridis
    newcolors = viridis(np.linspace(0, 1, 256))  # Get the RGBA values of these colors
    
    # Modify the last 10% of colors to be less bright, for example, a darker green
    last_colors = int(256 * 0.1)  # Modify the last 10% of the color range
    newcolors[-1:, :3] = np.array([0, 0.266, 0.105])  # A darker green (modify as needed)
    
    norm = mcolors.LogNorm(vmin=np.min(A[A > 0]), vmax=np.max(A))

    # Create a new colormap from the modified color data
    new_cmap = mcolors.ListedColormap(newcolors, name='custom_viridis')
    # Plot the resulting matrix
    plt.imshow(N, cmap=viridis,aspect='auto',interpolation='none')
    plt.colorbar(label='Firing frequency (Hz)')
    
    
#%%
import matplotlib.pyplot as plt
import numpy as np
import itertools
import matplotlib.colors as mcolors

def plotMultiD(A, param_names, row_params, col_params):
    dims = A.shape  # Get the dimensionality of A

    # Map param names to their indices
    param_indices = {name: idx for idx, name in enumerate(param_names)}

    # Get the indices for the row and column parameters
    row_indices = [param_indices[param] for param in row_params]
    col_indices = [param_indices[param] for param in col_params]

    # Calculate total number of rows and columns
    num_rows = np.prod([dims[i] for i in row_indices])
    num_cols = np.prod([dims[i] for i in col_indices])

    # Initialize the 2D matrix
    N = np.zeros((num_rows, num_cols))

    # Generate all combinations of indices for the given dimensions
    all_indices = list(itertools.product(*[range(d) for d in dims]))

    # Construct row and column index mappings
    def calculate_index(indices, order):
        index = 0
        for i, dim in enumerate(order):
            stride = np.prod([dims[order[j]] for j in range(i + 1, len(order))]) if i + 1 < len(order) else 1
            index += indices[dim] * stride
        return index

    # Evaluate the function for each combination of indices and populate the matrix
    for indices in all_indices:
        row_idx = calculate_index(indices, row_indices)
        col_idx = calculate_index(indices, col_indices)
        N[row_idx, col_idx] = A[indices]

    # Mask values outside the range 100 to 150
    masked_N = np.ma.masked_outside(N, 96, 156)
    # Create a custom colormap
    cmap = plt.cm.viridis  # Use a base colormap
    cmap.set_bad(color='dimgrey',alpha=0.2)  # Set masked values to red

    # Plot the resulting matrix
    plt.imshow(masked_N, cmap=cmap, aspect='auto', interpolation='none')
    plt.colorbar(label='Mean pause length (ms)')
    plt.show()

# Example usage
# Define A, param_names, row_params, col_params as per your data setup and call plotMultiD(...)
#%%
d=np.load('A_f28.npz')
low = 80
high = 100
count = np.sum((d['A'] >= low) & (d['A'] <= high))
print(f'Number of values within {low} and {high}: {count}')
#%%
#FIRST ONE PAUSE 6 PARAMS
d=np.load('A_p28.npz')
A[A > 400] = 400

plotMultiD(d['A'],param_names,['tauR','max_vCas','Iint'],['dres_coeff','Ca_half','vths']) 
plt.axis('off')
plt.tight_layout()
plt.savefig('cmapP.png',dpi=300)
plt.show()
#%%

#%%
plotMultiD(A,param_names,['tauR','Iint'],['vths','dres_coeff']) 
#%%
#FREQ 6 PARAMSSSSS
A[np.isnan(A)]=0
A[A > 500] = 500
plotMultiD(A,param_names,['g_sd','g0','Iint'],['g_ds','g1','eps_z0']) 
#plotMultiD(A, param_names, row_params, col_params):
#%%
#%%
d=np.load('A_f28.npz')

plotMultiD(d['A'],param_names,['g_sd','g0','Iint'],['g_ds','g1','eps_z0']) 
plt.axis('off')
plt.tight_layout()
#plt.savefig('cmapFphys.png',dpi=300)

#%%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

data = np.random.rand(10, 10) * 300  # Random data

# Define a custom colormap
cmap = plt.cm.viridis  # Use a base colormap
norm = mcolors.Normalize(vmin=np.min(data), vmax=np.max(data))  # Normalize data

# Highlight values between 100 and 200 using a mask
masked_data = np.ma.masked_outside(data, 100, 200)
cmap.set_bad(color='red')  # Set masked values to red

plt.figure()
plt.imshow(masked_data, cmap=cmap, norm=norm)
plt.colorbar()
plt.show()
#%%

#%%
import matplotlib.pyplot as plt
import numpy as np
import itertools
import matplotlib.colors as mcolors

def plotMultiD(A, param_names, row_params, col_params, param_ranges):
    dims = A.shape  # Get the dimensionality of A
    num_steps = A.shape[0]  # Assuming uniform step count across all dimensions

    # Prepare parameter values for mapping
    param_values = {param: np.linspace(start, stop, num_steps) for param, (start, stop) in param_ranges.items()}

    # Calculate row and column counts based on parameter indices
    num_rows = np.prod([dims[param_names.index(param)] for param in row_params])
    num_cols = np.prod([dims[param_names.index(param)] for param in col_params])

    fig, ax = plt.subplots()
    cmap = plt.cm.viridis
    cax = ax.imshow(A.reshape(num_rows, num_cols), aspect='auto', interpolation='none', cmap=cmap)
    fig.colorbar(cax)

    def on_click(event):
        # Calculate indices from the click position
        col_idx = int(event.xdata + 0.5)
        row_idx = int(event.ydata + 0.5)
        indices = np.unravel_index(row_idx * num_cols + col_idx, dims)  # Convert flat index to multidimensional index

        # Collect parameter values based on indices
        param_display = {param: param_values[param][indices[param_names.index(param)]] for param in param_names}
        print(f"Clicked on: {param_display}")

    cid = fig.canvas.mpl_connect('button_press_event', on_click)
    plt.show()

plotMultiD(d['A'],param_names,['Iint','g_sd','g0'],['eps_z0','g_ds','g1'],parameter_ranges) 
#%%
import matplotlib.pyplot as plt
import numpy as np
import itertools
import matplotlib.colors as mcolors

def plotMultiD(A, param_names, row_params, col_params, param_ranges):
    dims = A.shape  # Get the dimensionality of A

    # Prepare parameter values for mapping
    param_values = {param: np.linspace(start, stop, len(A[param_names.index(param)])) for param, (start, stop) in param_ranges.items()}

    # Find indices for row and column parameters in the order specified
    row_indices = [param_names.index(param) for param in row_params]
    col_indices = [param_names.index(param) for param in col_params]

    # Calculate the total number of rows and columns
    num_rows = np.prod([dims[idx] for idx in row_indices])
    num_cols = np.prod([dims[idx] for idx in col_indices])

    # Flatten the multi-dimensional array to a 2D array for plotting
    flat_indices = [param_names.index(param) for param in row_params + col_params]
    reshaped_A = A.transpose(flat_indices).reshape(num_rows, num_cols)

    fig, ax = plt.subplots()
    cmap = plt.cm.viridis
    cax = ax.imshow(reshaped_A, aspect='auto', interpolation='none', cmap=cmap)
    fig.colorbar(cax)

    def on_click(event):
        if event.inaxes == ax:
            col_idx = int(event.xdata + 0.5)
            row_idx = int(event.ydata + 0.5)
            indices = np.unravel_index(row_idx * num_cols + col_idx, dims)

            # Mapping clicked indices back to parameter values
            param_display = {}
            for param, idx in zip(row_params + col_params, indices):
                param_display[param] = param_values[param][idx]

            print(f"Clicked on: {param_display}")

    cid = fig.canvas.mpl_connect('button_press_event', on_click)
    plt.show()

# Example usage
plotMultiD(A, param_names, ['g_sd', 'g0', 'Iint'], ['g_ds', 'g1', 'eps_z0'], parameter_ranges)

