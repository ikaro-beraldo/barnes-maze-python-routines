# -*- coding: utf-8 -*-
"""
Created on Sun May 28 19:44:29 2023

@author: ikaro
"""

import numpy as np
import pandas as pd
import math
from matplotlib import pyplot as plt

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
filename = "E:\\Barnes Maze - Mestrad\\Parquinhos\\Final_results.h5"
save_results = "E:\\Barnes Maze - Mestrad\\Parquinhos\\"
trial_info = pd.read_hdf(filename, key='trial_info')  

for gg in np.arange(1,3,1):
    
    ################################### 1 - Latency for each trial throughout time
    # Loop to extract each day and trial separatelly
    latency_trial_av = np.zeros(16,)
    latency_trial_err = np.zeros(16,)
    x = []
    count = 0;
    
    for ii in np.arange(1,5,1):
        for jj in np.arange(1,5,1):
            # Create a temporary variable
            latency = trial_info.loc[(trial_info["Day"] == str(ii)) & (trial_info["Trial"] == str(jj)) & (trial_info["Group"] == str(gg)),['Latency']].to_numpy()
            # Get the average for each one of the day-trial combination (16)
            latency_trial_av[count] = np.mean(latency)
            # Get the standard deviation
            latency_trial_err[count] = np.std(latency)/math.sqrt(len(latency))
            # Add 1 to the iterator
            count += 1
            # Get the x axis for plot
            x.append('D'+str(ii)+'T'+str(jj))
    
    # plot Each Trial
    plt.figure(dpi=1200)
    plt.subplot(121)
    plt.errorbar(x, latency_trial_av, yerr=latency_trial_err)
    plt.xticks(rotation = 90)
    #axes.set_xlim((0,len(bp_pos_on_quadrant)/fps))
    plt.ylim((0,180))
    #axes.set_yticks(np.arange(1,5,1), ['#1','#2','#3','#4'])
    plt.xlabel("Days/Trials")
    plt.ylabel("Latency")
    plt.title("Average Latency for each trial")
    
    ###################################### 2 Latency for each Day
    latency_day_av = np.zeros(4,)
    latency_day_err = np.zeros(4,)
    x = []
    count = 0;
    
    for ii in np.arange(1,5,1):
        # Create a temporary variable
        latency = trial_info.loc[(trial_info["Day"] == str(ii)) & (trial_info["Group"] == str(gg)),['Latency']].to_numpy()
        # Get the average for each one of the day-trial combination (16)
        latency_day_av[count] = np.mean(latency)
        # Get the standard deviation
        latency_day_err[count] = np.std(latency)/math.sqrt(len(latency))
        # Add 1 to the iterator
        count += 1
        # Get the x axis for plot
        x.append('D'+str(ii))
            
    # plot Each Trial
    plt.subplot(122)
    plt.errorbar(x, latency_day_av, yerr=latency_day_err)
    plt.xticks(rotation = 90)
    #axes.set_xlim((0,len(bp_pos_on_quadrant)/fps))
    plt.ylim((np.min(latency_day_av)-10, np.max(latency_day_av)+10))
    #axes.set_yticks(np.arange(1,5,1), ['#1','#2','#3','#4'])
    plt.xlabel("Days")
    plt.ylabel("Latency")
    plt.title("Average Latency for each Day")
    
    plt.tight_layout()
    plt.savefig(save_results+str(gg)+"Average_latency.svg")
    
    
    ################################### 3 - Average speed for each trial throughout time
    # Loop to extract each day and trial separatelly
    speed_trial_av = np.zeros(16,)
    speed_trial_err = np.zeros(16,)
    x = []
    count = 0;
    
    for ii in np.arange(1,5,1):
        for jj in np.arange(1,5,1):
            # Create a temporary variable
            speed = trial_info.loc[(trial_info["Day"] == str(ii)) & (trial_info["Trial"] == str(jj)) & (trial_info["Group"] == str(gg)),['Av_speed']].to_numpy()
            # Get the average for each one of the day-trial combination (16)
            speed_trial_av[count] = np.mean(speed)
            # Get the standard deviation
            speed_trial_err[count] = np.std(speed)/math.sqrt(len(speed))
            # Add 1 to the iterator
            count += 1
            # Get the x axis for plot
            x.append('D'+str(ii)+'T'+str(jj))
    
    plt.figure()
    # plot Each Trial
    plt.figure(dpi=1200)
    plt.subplot(121)
    plt.errorbar(x, speed_trial_av, yerr=speed_trial_err)
    plt.xticks(rotation = 90)
    #axes.set_xlim((0,len(bp_pos_on_quadrant)/fps))
    plt.ylim((np.min(speed_trial_av)-np.std(speed_trial_av), np.max(speed_trial_av)+np.std(speed_trial_av)))
    #axes.set_yticks(np.arange(1,5,1), ['#1','#2','#3','#4'])
    plt.xlabel("Days/Trials")
    plt.ylabel("Speed cm/s")
    plt.title("Average Speed for each trial")
    
    ###################################### 4 speed for each Day
    speed_day_av = np.zeros(4,)
    speed_day_err = np.zeros(4,)
    x = []
    count = 0;
    
    for ii in np.arange(1,5,1):
        # Create a temporary variable
        speed = trial_info.loc[(trial_info["Day"] == str(ii)) & (trial_info["Group"] == str(gg)),['Av_speed']].to_numpy()
        # Get the average for each one of the day-trial combination (16)
        speed_day_av[count] = np.mean(speed)
        # Get the standard deviation
        speed_day_err[count] = np.std(speed)/math.sqrt(len(speed))
        # Add 1 to the iterator
        count += 1
        # Get the x axis for plot
        x.append('D'+str(ii))
            
    # plot Each Trial
    plt.subplot(122)
    plt.errorbar(x, speed_day_av, yerr=speed_day_err)
    plt.xticks(rotation = 90)
    #axes.set_xlim((0,len(bp_pos_on_quadrant)/fps))
    plt.ylim((np.min(speed_day_av)-np.std(speed_day_av), np.max(speed_day_av)+np.std(speed_day_av)))
    #axes.set_yticks(np.arange(1,5,1), ['#1','#2','#3','#4'])
    plt.xlabel("Days")
    plt.ylabel("Speed cm/s")
    plt.title("Average Speed for each Day")
    plt.savefig(save_results+str(gg)+"Average_Speed.svg")
    
    
    plt.tight_layout()
    
    ################################### 5 - primary and secundary errors each trial throughout time
    # Loop to extract each day and trial separatelly
    p_err_trial_av = np.zeros(16,)
    p_err_trial_err = np.zeros(16,)
    s_err_trial_av = np.zeros(16,)
    s_err_trial_err = np.zeros(16,)
    x = []
    count = 0;
    
    for ii in np.arange(1,5,1):
        for jj in np.arange(1,5,1):
            
            # PRIMARY ERROR
            # Create a temporary variable
            p_err = trial_info.loc[(trial_info["Day"] == str(ii)) & (trial_info["Trial"] == str(jj)) & (trial_info["Group"] == str(gg)),['P_error']].to_numpy()
            # Get the average for each one of the day-trial combination (16)
            p_err_trial_av[count] = np.mean(p_err)
            # Get the standard deviation
            p_err_trial_err[count] = np.std(p_err)/math.sqrt(len(p_err))
            
            # SECUNDARY ERROR
            # Create a temporary variable 
            s_err = trial_info.loc[(trial_info["Day"] == str(ii)) & (trial_info["Trial"] == str(jj)),['S_error']].to_numpy()
            # Get the average for each one of the day-trial combination (16)
            s_err_trial_av[count] = np.mean(s_err)
            # Get the standard deviation
            s_err_trial_err[count] = np.std(s_err)/math.sqrt(len(s_err))
            
            # Add 1 to the iterator
            count += 1
            # Get the x axis for plot
            x.append('D'+str(ii)+'T'+str(jj))
    
    plt.figure()
    # plot Each Trial
    plt.figure(dpi=1200)
    plt.subplot(121)
    plt.errorbar(x, p_err_trial_av, yerr=p_err_trial_err, label='Primary error')
    plt.errorbar(x, s_err_trial_av, yerr=s_err_trial_err, label='Secundary error')
    plt.xticks(rotation = 90)
    #axes.set_xlim((0,len(bp_pos_on_quadrant)/fps))
    #plt.ylim((np.min(speed_trial_av)-np.std(speed_trial_av), np.max(speed_trial_av)+np.std(speed_trial_av)))
    #axes.set_yticks(np.arange(1,5,1), ['#1','#2','#3','#4'])
    plt.xlabel("Days/Trials")
    plt.ylabel("Errors")
    plt.title("Number of Errors each Trial")
    plt.legend()
    
    ###################################### 6 Primary and secundary errors for each Day
    p_err_day_av = np.zeros(4,)
    p_err_day_err = np.zeros(4,)
    s_err_day_av = np.zeros(4,)
    s_err_day_err = np.zeros(4,)
    x = []
    count = 0;
    
    for ii in np.arange(1,5,1):
        # PRIMARY ERROR
        # Create a temporary variable
        p_err = trial_info.loc[(trial_info["Day"] == str(ii)) & (trial_info["Group"] == str(gg)),['P_error']].to_numpy()
        # Get the average for each one of the day-trial combination (16)
        p_err_day_av[count] = np.mean(p_err)
        # Get the standard deviation
        p_err_day_err[count] = np.std(p_err)/math.sqrt(len(p_err))
        
        # SECUNDARY ERROR
        # Create a temporary variable 
        s_err = trial_info.loc[(trial_info["Day"] == str(ii)) & (trial_info["Group"] == str(gg)),['S_error']].to_numpy()
        # Get the average for each one of the day-trial combination (16)
        s_err_day_av[count] = np.mean(s_err)
        # Get the standard deviation
        s_err_day_err[count] = np.std(s_err)/math.sqrt(len(s_err))
        
        # Add 1 to the iterator
        count += 1
        # Get the x axis for plot
        x.append('D'+str(ii))
            
    # plot Each Trial
    plt.subplot(122)
    plt.errorbar(x, p_err_day_av, yerr=p_err_day_err, label='Primary error')
    plt.errorbar(x, s_err_day_av, yerr=s_err_day_err, label='Secundary error')
    plt.xticks(rotation = 90)
    #axes.set_xlim((0,len(bp_pos_on_quadrant)/fps))
    #plt.ylim((np.min(speed_day_av)-np.std(speed_day_av), np.max(speed_day_av)+np.std(speed_day_av)))
    #axes.set_yticks(np.arange(1,5,1), ['#1','#2','#3','#4'])
    plt.xlabel("Days")
    plt.ylabel("Errors")
    plt.title("Number of Errors for each Day")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_results+str(gg)+"Average_Errors.svg")
    
    
    
    ################################### 7 - Average distance for each trial throughout time
    # Loop to extract each day and trial separatelly
    distance_trial_av = np.zeros(16,)
    distance_trial_err = np.zeros(16,)
    x = []
    count = 0;
    
    for ii in np.arange(1,5,1):
        for jj in np.arange(1,5,1):
            # Create a temporary variable
            distance = trial_info.loc[(trial_info["Day"] == str(ii)) & (trial_info["Trial"] == str(jj)) & (trial_info["Group"] == str(gg)),['Distance']].to_numpy()
            # Get the average for each one of the day-trial combination (16)
            distance_trial_av[count] = np.mean(distance)
            # Get the standard deviation
            distance_trial_err[count] = np.std(distance)/math.sqrt(len(distance))
            # Add 1 to the iterator
            count += 1
            # Get the x axis for plot
            x.append('D'+str(ii)+'T'+str(jj))
    
    plt.figure()
    # plot Each Trial
    plt.figure(dpi=1200)
    plt.subplot(121)
    plt.errorbar(x, distance_trial_av, yerr=distance_trial_err)
    plt.xticks(rotation = 90)
    #axes.set_xlim((0,len(bp_pos_on_quadrant)/fps))
    plt.ylim((np.min(distance_trial_av)-np.std(distance_trial_av), np.max(distance_trial_av)+np.std(distance_trial_av)))
    #axes.set_yticks(np.arange(1,5,1), ['#1','#2','#3','#4'])
    plt.xlabel("Days/Trials")
    plt.ylabel("Distance")
    plt.title("Average Distance for each trial")
    
    ###################################### 8 distance for each Day
    distance_day_av = np.zeros(4,)
    distance_day_err = np.zeros(4,)
    x = []
    count = 0;
    
    for ii in np.arange(1,5,1):
        # Create a temporary variable
        distance = trial_info.loc[(trial_info["Day"] == str(ii)) & (trial_info["Group"] == str(gg)),['Distance']].to_numpy()
        # Get the average for each one of the day-trial combination (16)
        distance_day_av[count] = np.mean(distance)
        # Get the standard deviation
        distance_day_err[count] = np.std(distance)/math.sqrt(len(distance))
        # Add 1 to the iterator
        count += 1
        # Get the x axis for plot
        x.append('D'+str(ii))
            
    # plot Each Trial
    plt.subplot(122)
    plt.errorbar(x, distance_day_av, yerr=distance_day_err)
    plt.xticks(rotation = 90)
    #axes.set_xlim((0,len(bp_pos_on_quadrant)/fps))
    plt.ylim((np.min(distance_day_av)-np.std(distance_day_av), np.max(distance_day_av)+np.std(distance_day_av)))
    #axes.set_yticks(np.arange(1,5,1), ['#1','#2','#3','#4'])
    plt.xlabel("Days")
    plt.ylabel("Distance cm")
    plt.title("Average Distance for each Day")
    
    plt.tight_layout()
    plt.savefig(save_results+str(gg)+"Average_Distance.svg")
    
    
    
    ################################### 9 - Strategy used for each trial throughout time
    # Loop to extract each day and trial separatelly
    spatial_trial_av = np.zeros(16,)
    serial_trial_av = np.zeros(16,)
    random_trial_av = np.zeros(16,)
    
    x = []
    count = 0;
    
    for ii in np.arange(1,5,1):
        for jj in np.arange(1,5,1):
            # Create a temporary variable
            strategy = trial_info.loc[(trial_info["Day"] == str(ii)) & (trial_info["Trial"] == str(jj)) & (trial_info["Group"] == str(gg)),['Strategy']].to_numpy()
            # Get the number of times (proportion) of each strategy for each one of the day-trial combination (16)
            spatial_trial_av[count] = len(np.where(strategy=='spatial')[0])/len(strategy)
            serial_trial_av[count] = len(np.where(strategy=='serial')[0])/len(strategy)
            random_trial_av[count] = len(np.where(strategy=='random')[0])/len(strategy)
            # Add 1 to the iterator
            count += 1
            # Get the x axis for plot
            x.append('D'+str(ii)+'T'+str(jj))
    
    # prepare data for plotting
    weight_counts = dict({"Random":random_trial_av,
                          "Serial":serial_trial_av,
                          "Spatial":spatial_trial_av})
    
    figure, axes = plt.subplots(2,1, dpi=1200)
    
    plot_stacked_bar(tuple(x), weight_counts, 3, ax=axes[0])
    #axes[0,0].set_xticks(ticks=x, rotation = 90)
    #axes.set_xlim((0,len(bp_pos_on_quadrant)/fps))
    axes[0].set_ylim((0,1.2))
    #axes.set_yticks(np.arange(1,5,1), ['#1','#2','#3','#4'])
    axes[0].set_xlabel("Days/Trials")
    axes[0].set_ylabel("% Strategy used")
    #axes[0].set_title("Strategy use by Trial")
    
    
    # ###################################### 8 distance for each Day
    spatial_day_av = np.zeros(4,)
    serial_day_av = np.zeros(4,)
    random_day_av = np.zeros(4,)
    x = []
    count = 0;
    
    for ii in np.arange(1,5,1):
        # Create a temporary variable
        strategy = trial_info.loc[(trial_info["Day"] == str(ii)) & (trial_info["Group"] == str(gg)),['Strategy']].to_numpy()
        # Get the number of times (proportion) of each strategy for each one of the day-trial combination (16)
        spatial_day_av[count] = len(np.where(strategy=='spatial')[0])/len(strategy)
        serial_day_av[count] = len(np.where(strategy=='serial')[0])/len(strategy)
        random_day_av[count] = len(np.where(strategy=='random')[0])/len(strategy)
        # Add 1 to the iterator
        count += 1
        # Get the x axis for plot
        x.append('D'+str(ii))
        
    # prepare data for plotting
    weight_counts = dict({"Random":random_day_av,
                          "Serial":serial_day_av,
                          "Spatial":spatial_day_av})
            
    # plot Each Trial
    plot_stacked_bar(tuple(x), weight_counts, 3, width=0.25, ax=axes[1])
    plt.tight_layout()
    #axes[0,0].set_xticks(ticks=x, rotation = 90)
    #axes.set_xlim((0,len(bp_pos_on_quadrant)/fps))
    axes[1].set_ylim((0,1.2))
    #axes.set_yticks(np.arange(1,5,1), ['#1','#2','#3','#4'])
    axes[1].set_xlabel("Days")
    axes[1].set_ylabel("% Strategy used")
    #axes[1].set_title("Strategy use by Day")
    
    plt.tight_layout()
    plt.savefig(save_results+str(gg)+"Average_Strategy.svg")



