# -*- coding: utf-8 -*-
"""
Created on Fri May 26 12:21:12 2023

@author: ikaro

Routine to run all the analysis to all the trials
"""


import pandas as pd
import numpy as np 
from tkinter import Tk
import csv
import math
import matplotlib
from matplotlib import pyplot as plt
from scipy import stats as sts
from data_handling import *
from maze_modeling import *
from extract_frame import *
from data_processing import *
from plot_functions import *
from data_analysis import *
import cv2
from gif_functions import *
import os
import json

def main_batch_run():

    # SET parameter values
    conf_threshold = 0.95
    std_threshold = 0.5
    fps = 30
    prev_hole_position = np.empty((12,2))
    
    # CREATE A DATA FRAME TO ORGANIZE THE RSULTS FOR ALL THE TRIALS
    trial_info = pd.DataFrame(columns=['Box','ID','Group','Day','Trial','Beginning', 'End', 'Latency', 'P_error', 'S_error',
                                       'Strategy', 'Distance', 'Av_speed'])
    
    # STEP 1 --> SELECT THE MULTIPLE FILES
    filename = select_file(multiple=True)
    
    # STEP 2 --> LOOP FOR EACH FILE
    for it in range(len(filename)):
                
        # STEP 3 --> LOAD FILE AND RECREATE THE MAZE
        # Load the file
        df = pd.read_hdf(filename[it])
        # Get each hole position
        position_each_hole = get_hole_positions(df, prev_hole_position=prev_hole_position) 
        prev_hole_position = position_each_hole # UPDATE THE HOLE POSITION IN CASE OF AN ERROR
        # Get the maze centroid
        centroid_coords = centroid_inference(position_each_hole)    
        # Get the maze coordinates in pixel
        maze_info_pixel = maze_recreation(position_each_hole, 10, 5, 60)
        # Recreate the maze quadrants
        quadrant_dict, quadrant_dict_list = maze_quadrants(maze_info_pixel, [], centroid_coords, position_each_hole, plot_frame=False, title='Nose', show=True, recreate_maze=False)
        
        
        # STEP 3.1 --> GET THE NOSE COORD AND PROCESS IT
        # Get the coordinates for a specific body part
        nose_coord = df.xs('nose', level='bodyparts', axis=1).to_numpy()  
        # Fix coordinates inconsistencies based on confidence interval
        body_part_matrix_nose = fix_frames_confidence(nose_coord,conf_threshold)  
        ## OLD FUNCTION --> Fix coordinates inconsistencies based on (x,y) diff standard deviation
        # body_part_matrix_nose = fix_frames_diff(body_part_matrix_nose,std_threshold)
        
        # STEP 3.2 --> GET THE TRIAL BEGINNING, END AND LATENCY
        # Define the trial beginning and end based on confidence interval
        body_part_matrix_nose, beg, end = get_trial_beginning_end_all_bp(body_part_matrix_nose, df, 0.95)
        # OLD FUNCTION -->> body_part_matrix_nose = get_trial_beginning_end(body_part_matrix_nose, 0.95)
        # Get trial latency
        latency = get_trial_latency(body_part_matrix_nose, fps=30)      
        
        # STEP 4 --> CREATE A CODE FOR THE NOSE POSITION ON MAZE
        # Get the nose position on the maze
        bp_pos_on_maze = get_bp_position_on_maze(body_part_matrix_nose, position_each_hole, maze_info_pixel, centroid_coords)
        # Filter the nose position
        bp_pos_on_maze_filtered = filter_bp_pos_on_maze(bp_pos_on_maze, method_used='complete', win=fps)
        
        # STEP 5 --> GET PRIMARY AND SECUNDARY ERRORS and STRATEGY USED
        p_errors, s_errors = get_p_s_errors(bp_pos_on_maze_filtered,target=1)[0:2]
        strategy = get_the_strategy(bp_pos_on_maze_filtered, target=1)
        
        # STEP 6.1 --> GET THE BODY CENTRE COORD AND PROCESS IT
        # Get the coordinates for a specific body part
        body_centre_coord = df.xs('body_centre', level='bodyparts', axis=1).to_numpy()  
        # Fix coordinates inconsistencies based on confidence interval
        body_part_matrix_body_centre = fix_frames_confidence(body_centre_coord,conf_threshold)
        
        # UPDATE the beginning and end for the body_centre as well
        body_part_matrix_body_centre, beg, end = get_trial_beginning_end_all_bp(body_part_matrix_body_centre, df, 0.95)
        
        # STEP 7 --> GET THE TOTAL DISTANCE, INSTANT SPEED AND AVERAGE SPEED
        total_distance = get_distance(body_part_matrix_body_centre, maze_info_pixel)[1]
        inst_speed, inst_speed_entire, av_speed = get_inst_speed(body_part_matrix_body_centre, maze_info_pixel, time_win=10, fps=30)
            
        # STEP 7.1 --> BODY PART POSITION ON QUADRANT
        bp_pos_on_quadrant = get_bp_position_on_quadrant(body_part_matrix_body_centre, quadrant_dict, fps=30)
        
        # STEP 7.2 --> RATIO (TIME ON TARGET/ TIME ON OTHER QUADRANTS)
        ratio_target_others, time_on_each_quadrant = get_time_on_target_quadrant(bp_pos_on_quadrant, target=1, fps=30)    
        
        # STEP 8 --> ORGANIZE DATA SINCE NOT EVERY SINGLE THING IS BAGUNÇA
        # Get the file name (only the base name)
        basename = list(os.path.basename(filename[it]))
        box = ''.join(basename[0:3])
        ID = basename[4]
        group = basename[7]
        day = basename[10]
        trial = basename[13]
        
        # Create a data frame to append to the final dataframe
        data = pd.DataFrame([[box, ID, group, day, trial, beg, end, latency, p_errors, s_errors, strategy, total_distance, av_speed, ratio_target_others]], 
                            columns = ['Box','ID','Group','Day','Trial','Beginning', 'End', 'Latency', 'P_error', 'S_error',
                                           'Strategy', 'Distance', 'Av_speed','Time_on_target']) 
        # makes index continuous
        trial_info = pd.concat([trial_info, data], ignore_index = True)  
        
        # Trial temporal series
        trial_temp_series = dict({'bp_pos_on_maze':bp_pos_on_maze.tolist(), 
                                  'bp_pos_on_maze_filtered': bp_pos_on_maze_filtered.tolist(),
                                  'bp_pos_on_quadrant':bp_pos_on_quadrant.tolist(),
                                  'body_part_matrix_body_centre':body_part_matrix_body_centre.tolist(),
                                  'body_part_matrix_nose':body_part_matrix_nose.tolist(),
                                  'position_each_hole':position_each_hole.tolist(),
                                  'centroid_coords':centroid_coords.tolist(),
                                  'maze_info_pixel':maze_info_pixel,
                                  'quadrant_dict_list':quadrant_dict_list,
                                  'total_distance':total_distance.tolist(),
                                  'inst_speed':inst_speed.tolist(),
                                  'inst_speed_entire':inst_speed_entire.tolist(),
                                  'time_on_each_quadrant':time_on_each_quadrant.tolist()})
        
        # Get the name to save a file
        save_filename = os.path.dirname(filename[it])+'/'+''.join(basename[0:14])+'.txt'
        # Save temporal series as a json file
        with open(save_filename, "w", encoding='utf-8') as fp:
            json.dump(trial_temp_series, fp, indent=4)  # encode dict into JSON
       
        # STEP 9 --> PLOT WITH THE MAIN INFORMATIONS
        save_filename_plot = os.path.dirname(filename[it])+'/'+''.join(basename[0:14])
        big_plot(body_part_matrix_nose, body_part_matrix_body_centre, bp_pos_on_maze_filtered, centroid_coords, position_each_hole, maze_info_pixel, inst_speed_entire, quadrant_dict, bp_pos_on_quadrant, save_filename_plot, trial_data=data, trial_name=''.join(basename[0:14]), show=False)
         
        # STEP 10 --> CREATE A COOL GIF (SLOW.....)
        #create_gif(body_part_matrix, bp_pos_on_maze, centroid_coords, position_each_hole, maze_info_pixel, inst_speed_entire, quadrant_dict, bp_pos_on_quadrant, plot_frame=False, title='Nose', show=False)
        
        # Lil' print to inform to the user that this specific trial has been analysed
        print(str(it)+' - '+''.join(basename[0:14]) + str(': OK!'))
        
    # STEP 11 --> Save the final dataframe as json
    save_filename = os.path.dirname(filename[it])+'/'+'Final_results'+'.h5'
    trial_info.to_hdf(save_filename, key='trial_info', mode='w')  
    
    trial_info = pd.read_hdf(save_filename, key='trial_info')  