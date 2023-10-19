# -*- coding: utf-8 -*-
"""
Created on Sun May 28 19:44:29 2023

@author: ikaro
"""

import numpy as np
import pandas as pd
import math
from matplotlib import pyplot as plt
from scipy.io import savemat

def plot_stacked_bar(x, weight_counts, n_elements, width=0.5, ax=None):    
   
    # EXAMPLE
    # x = (
    #     "Adelie\n $\\mu=$3700.66g",
    #     "Chinstrap\n $\\mu=$3733.09g",
    #     "Gentoo\n $\\mu=5076.02g$",
    # )
    # weight_counts = {
    #     "Below": np.array([70, 31, 58]),
    #     "Above": np.array([82, 37, 66]),
    # }
       
    if ax is None:
        fig, ax = plt.subplots()
        
    bottom = np.zeros(len(x))
    
    for ii in range(n_elements):
        p = ax.bar(x, list(weight_counts.values())[ii], width, label=list(weight_counts)[ii], bottom=bottom)
        bottom += list(weight_counts.values())[ii]
    
    ax.legend(loc="upper center", ncol=n_elements, bbox_to_anchor=(0, 1.1, 1, 0.2), frameon=False)
    ax.set_xticks(ticks=x, labels=x, rotation=90)


# Open file
filename = "E:\\Barnes Maze - Mestrad\\Parquinhos\\ProbTrial\\Final_results.h5"
save_results = "E:\\Barnes Maze - Mestrad\\Parquinhos\\ProbTrial\\"
trial_info = pd.read_hdf(filename, key='trial_info')  
prob = True


# Get the unique box/ID IDs
trial_info["ID_unique"] = 0
box = trial_info["Box"].unique()
# Get the animal IDS
IDs = trial_info["ID"].unique()
id_unique = 0
for ii in np.arange(0,len(box),1):
    for jj in np.arange(0,len(IDs),1):
        trial_info["ID_unique"].loc[(trial_info["Box"] == box[ii]) & (trial_info["ID"] == IDs[jj])] = id_unique
        if ((trial_info["Box"] == box[ii]) & (trial_info["ID"] == IDs[jj])).any():
            id_unique = id_unique + 1
len(IDs)

# Create a trial backup
#trial_info_backup = trial_info

# Get the new unique IDs
IDs = trial_info["ID_unique"].unique()

for gg in np.arange(1,3,1):
    
    # Get only the trials regarding the group
    #trial_info = trial_info_backup.loc[(trial_info_backup["Group"] == str(gg))]
    IDs = trial_info["ID_unique"].loc[(trial_info["Group"] == str(gg))].unique()

    ################################### 1 - Latency for each trial throughout time
    # Loop to extract each day and trial separatelly
    latency_trial_av = np.zeros((4,len(IDs)))
    latency_trial_err = np.zeros((4,len(IDs)))
    x = []
    count = 0;
    
    for ii in np.arange(1,2,1):
        for jj in np.arange(0,len(IDs),1):
            # Create a temporary variable
            latency = trial_info.loc[(trial_info["ID_unique"] == IDs[jj]) & (trial_info["Group"] == str(gg)),['Latency']].to_numpy()
            # Get the average for each one of the day-trial combination (16)
            latency_trial_av[ii-1,jj] = np.mean(latency)
            # Get the standard deviation
            latency_trial_err[ii-1,jj] = np.std(latency)/math.sqrt(len(latency))
            # Add 1 to the iterator
            count += 1
            # Get the x axis for plot
            x.append('D'+str(ii)+'T'+str(IDs[jj]))
    
    
    
    ################################### 3 - Average speed for each trial throughout time
    # Loop to extract each day and trial separatelly
    speed_trial_av = np.zeros((4,len(IDs)))
    speed_trial_err = np.zeros((4,len(IDs)))
    x = []
    count = 0;
    
    for ii in np.arange(1,2,1):
        for jj in np.arange(0,len(IDs),1):
            # Create a temporary variable
            speed = trial_info.loc[(trial_info["ID_unique"] == IDs[jj]) & (trial_info["Group"] == str(gg)),['Av_speed']].to_numpy()
            # Get the average for each one of the day-trial combination (16)
            speed_trial_av[ii-1,jj] = np.mean(speed)
            # Get the standard deviation
            speed_trial_err[ii-1,jj] = np.std(speed)/math.sqrt(len(speed))
            # Add 1 to the iterator
            count += 1
            # Get the x axis for plot
            x.append('D'+str(ii)+'T'+str(IDs[jj]))
    
    
    ################################### 5 - primary and secundary errors each trial throughout time
    # Loop to extract each day and trial separatelly
    p_err_trial_av = np.zeros((4,len(IDs)))
    p_err_trial_sum = np.zeros((4,len(IDs)))
    p_err_trial_err = np.zeros((4,len(IDs)))
    s_err_trial_av = np.zeros((4,len(IDs)))
    s_err_trial_sum = np.zeros((4,len(IDs)))
    s_err_trial_err = np.zeros((4,len(IDs)))
    x = []
    count = 0;
    
    for ii in np.arange(1,2,1):
        for jj in np.arange(0,len(IDs),1):
            
            # PRIMARY ERROR
            # Create a temporary variable
            p_err = trial_info.loc[(trial_info["ID_unique"] == IDs[jj]) & (trial_info["Group"] == str(gg)),['P_error']].to_numpy()
            # Get the average for each one of the day-trial combination (16)
            p_err_trial_av[ii-1,jj] = np.mean(p_err)
            # Get the sum for each one of the day-trial combination (16)
            p_err_trial_sum[ii-1,jj] = np.sum(p_err)
            # Get the standard deviation
            p_err_trial_err[ii-1,jj] = np.std(p_err)/math.sqrt(len(p_err))
            
            # SECUNDARY ERROR
            # Create a temporary variable 
            s_err = trial_info.loc[(trial_info["ID_unique"] == IDs[jj]) & (trial_info["Group"] == str(gg)),['S_error']].to_numpy()
            # Get the average for each one of the day-trial combination (16)
            s_err_trial_av[ii-1,jj] = np.mean(s_err)
            # Get the sum for each one of the day-trial combination (16)
            s_err_trial_sum[ii-1,jj] = np.sum(s_err)
            # Get the standard deviation
            s_err_trial_err[ii-1,jj] = np.std(s_err)/math.sqrt(len(s_err))
            
            # Add 1 to the iterator
            count += 1
            # Get the x axis for plot
            x.append('D'+str(ii)+'T'+str(IDs[jj]))
    
    
    
    ################################### 7 - Average distance for each trial throughout time
    # Loop to extract each day and trial separatelly
    distance_trial_av = np.zeros((4,len(IDs)))
    distance_trial_err = np.zeros((4,len(IDs)))
    x = []
    count = 0;
    
    for ii in np.arange(1,2,1):
        for jj in np.arange(0,len(IDs),1):
            # Create a temporary variable
            distance = trial_info.loc[(trial_info["ID_unique"] == IDs[jj]) & (trial_info["Group"] == str(gg)),['Distance']].to_numpy()
            # Get the average for each one of the day-trial combination (16)
            distance_trial_av[ii-1,jj] = np.mean(distance)
            # Get the standard deviation
            distance_trial_err[ii-1,jj] = np.std(distance)/math.sqrt(len(distance))
            # Add 1 to the iterator
            count += 1
            # Get the x axis for plot
            x.append('D'+str(ii)+'T'+str(IDs[jj]))
            
    
    ################################### 8 - Average Time on target for each trial throughout time
    # Loop to extract each day and trial separatelly
    time_on_target_trial_av = np.zeros((4,len(IDs)))
    time_on_target_trial_err = np.zeros((4,len(IDs)))
    x = []
    count = 0;
    
    for ii in np.arange(1,2,1):
        for jj in np.arange(0,len(IDs),1):
            # Create a temporary variable
            time_on_target = trial_info.loc[(trial_info["ID_unique"] == IDs[jj]) & (trial_info["Group"] == str(gg)),['Time_on_target']].to_numpy()
            # Get the average for each one of the day-trial combination (16)
            time_on_target_trial_av[ii-1,jj] = np.mean(time_on_target)
            # Get the standard deviation
            time_on_target_trial_err[ii-1,jj] = np.std(time_on_target)/math.sqrt(len(time_on_target))
            # Add 1 to the iterator
            count += 1
            # Get the x axis for plot
            x.append('D'+str(ii)+'T'+str(IDs[jj]))
    
     
    
    ################################### 9 - Strategy used for each trial throughout time
    # Loop to extract each day and trial separatelly
    spatial_trial_av = np.zeros((4,len(IDs)))
    serial_trial_av = np.zeros((4,len(IDs)))
    random_trial_av = np.zeros((4,len(IDs)))
    spatial_trial_sum = np.zeros((4,len(IDs)))
    serial_trial_sum = np.zeros((4,len(IDs)))
    random_trial_sum = np.zeros((4,len(IDs)))
    
    x = []
    count = 0;
    
    for ii in np.arange(1,2,1):
        for jj in np.arange(0,len(IDs),1):
            # Create a temporary variable
            strategy = trial_info.loc[(trial_info["ID_unique"] == IDs[jj]) & (trial_info["Group"] == str(gg)),['Strategy']].to_numpy()
            # Get the number of times (proportion) of each strategy for each one of the day-trial combination (16)
            spatial_trial_av[ii-1,jj] = len(np.where(strategy=='spatial')[0])/len(strategy)
            serial_trial_av[ii-1,jj] = len(np.where(strategy=='serial')[0])/len(strategy)
            random_trial_av[ii-1,jj] = len(np.where(strategy=='random')[0])/len(strategy)
            spatial_trial_sum[ii-1,jj] = len(np.where(strategy=='spatial')[0])
            serial_trial_sum[ii-1,jj] = len(np.where(strategy=='serial')[0])
            random_trial_sum[ii-1,jj] = len(np.where(strategy=='random')[0])
            # Add 1 to the iterator
            count += 1
            # Get the x axis for plot
            x.append('D'+str(ii)+'T'+str(IDs[jj]))
    
    # Get the ratio target exploration minute1/minute2
    min1_min2 = np.zeros(len(IDs),)
    if prob is True:
        
        min1_min2_total = trial_info.loc[(trial_info["Group"] == str(gg)),["Time_on_target_minute"]].to_numpy()
        for ii in range(len(min1_min2_total)):
            min1_min2[ii] = min1_min2_total[ii][0][0] / min1_min2_total[ii][0][1]
    
    
    
    ################################### Save as mat-file
    
    mdic = {"distance_trial_av": distance_trial_av, 
            "latency_trial_av": latency_trial_av,
            "p_err_trial_av": p_err_trial_av,
            "s_err_trial_av":s_err_trial_av,
            "p_err_trial_sum": p_err_trial_sum,
            "s_err_trial_sum":s_err_trial_sum,
            "speed_trial_av":speed_trial_av,
            "time_on_target_trial_av":time_on_target_trial_av,
            "spatial_trial_av":spatial_trial_av,
            "random_trial_av":random_trial_av,
            "serial_trial_av":serial_trial_av,
            "spatial_trial_sum":spatial_trial_sum,
            "random_trial_sum":random_trial_sum,
            "serial_trial_sum":serial_trial_sum,
            "min1_min2":min1_min2}
    
    mdic_norm = {"distance_trial_av": distance_trial_av / distance_trial_av[0,:], 
            "latency_trial_av": latency_trial_av / latency_trial_av[0,:],
            "p_err_trial_av": p_err_trial_av,
            "s_err_trial_av":s_err_trial_av,
            "p_err_trial_sum": p_err_trial_sum / p_err_trial_sum[0,:],
            "s_err_trial_sum":s_err_trial_sum / s_err_trial_sum[0,:],
            "speed_trial_av":speed_trial_av / speed_trial_av[0,:],
            "time_on_target_trial_av":time_on_target_trial_av / time_on_target_trial_av[0,:],
            "spatial_trial_av":spatial_trial_av,
            "random_trial_av":random_trial_av,
            "serial_trial_av":serial_trial_av,
            "spatial_trial_sum":spatial_trial_sum,
            "random_trial_sum":random_trial_sum,
            "serial_trial_sum":serial_trial_sum,
            "min1_min2":min1_min2}
    
    savemat(save_results+"barnes_maze_result_matrix"+str(gg)+".mat", mdic_norm)
