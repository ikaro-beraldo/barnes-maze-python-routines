# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 15:01:33 2023

@author: ikaro
"""

# Set of functions regarding the Barnes Maze (pixel/cm calc; centroid inference; Maze recreation)

import numpy as np
from matplotlib import pyplot as plt
from plot_functions import *
from minors_functions import linear_function
import math

def get_vertex_positions(dlc_data_frame, bp_names, prev_vertex_position=None):
    min_confidence = 0.99    # Minimum confidence threshold
    # Create an empty matrix for the vertices coordinates
    position_each_vertex = np.zeros((8,2))
    # Loop for each one the 12 vertices
    for i in range(8):
        # convert the vertex x,y and confidence values to numpy array
        vertex = dlc_data_frame.xs(bp_names['v_'+str(i+1)], level='bodyparts', axis=1).to_numpy()
        # compare the confidence for each frame 
        confidence_mask = np.where(vertex[:,2] >= min_confidence)
        # average the (x,y) coords for only the frames that have the min confidence
        position_each_vertex[i,:] = np.array((np.average(vertex[confidence_mask,0]), np.average(vertex[confidence_mask,1])))
    
    # Check if all vertices got valid coords values, if not, get a prev_vertex_position
    if np.sum(np.isnan(position_each_vertex)) != 0:
        r,c = np.where(np.isnan(position_each_vertex)==True)  # Get the indices where it is NaN
        for i,j in zip(r,c): # Loop for each NaN value
            position_each_vertex[i,j] = prev_vertex_position[i,j]
        
    
    return position_each_vertex

# Calculate the Maze centroid based on the 12 vertices coordinates
def centroid_inference(vertex_coords):
    x = np.sum(vertex_coords[:,0])/len(vertex_coords[:,0])
    y = np.sum(vertex_coords[:,1])/len(vertex_coords[:,1])
    
    return np.array((x,y))

# Recreate the maze area accordingly to real and pixel measurements
def maze_recreation(maze_info_pixels, vertex_coords, distance_from_edge, vertex_radius, maze_radius):
    # Get the distance between the center of vertexs 1 to 7
    # real_dist_L = 67 # Real distance of the long side in cm
    # real_dist_S = 50 # Real distance of the short side in cm
    
    real_dist_L = maze_info_pixels['box_length'][0] # Real distance of the long side in cm
    real_dist_S = maze_info_pixels['box_length'][1] # Real distance of the short side in cm
    #print('real dist vertexs =' + str(real_dist_vertexs))
    pixel_dist_vertexs_L = np.empty((2,1))
    pixel_dist_vertexs_S = np.empty((2,1))
    
    # Calculate the difference between vertices
    pixel_dist_vertexs_L[0] = np.sqrt((vertex_coords[0,0]-vertex_coords[1,0])**2 + (vertex_coords[0,1]-vertex_coords[1,1])**2) # Pixel distance
    pixel_dist_vertexs_L[1] = np.sqrt((vertex_coords[2,0]-vertex_coords[3,0])**2 + (vertex_coords[2,1]-vertex_coords[3,1])**2) # Pixel distance
    pixel_dist_vertexs_S[0] = np.sqrt((vertex_coords[0,0]-vertex_coords[3,0])**2 + (vertex_coords[0,1]-vertex_coords[3,1])**2) # Pixel distance
    pixel_dist_vertexs_S[1] = np.sqrt((vertex_coords[1,0]-vertex_coords[2,0])**2 + (vertex_coords[1,1]-vertex_coords[2,1])**2) # Pixel distance
    
    # Average it
    pixel_dist_vertexs_L = np.mean(pixel_dist_vertexs_L)
    pixel_dist_vertexs_S = np.mean(pixel_dist_vertexs_S)
    #print('final:' + str(pixel_dist_vertexs))
    
    # Create variables with important parameters
    pixelcm_ratio = np.mean(np.array((pixel_dist_vertexs_L/real_dist_L, pixel_dist_vertexs_S/real_dist_S)))
       
    # Produce a dict with the maze elements PIXEL length values
    #maze_info_pixels = dict({'pixelcm_ratio': pixelcm_ratio})
    maze_info_pixels['pixelcm_ratio'] = pixelcm_ratio
    
    return maze_info_pixels


# Function to define the maze quadrants
def maze_quadrants(maze_info_pixel, body_part_matrix, centroid_coords, position_each_vertex, plot_frame=False, title='Nose', show=True, recreate_maze=False):
    # Get the points between the exterior vertices 5, 6, 7, 8
    
    # Pre-allocate the x and y vectors
    x = np.empty((4,))
    y = np.empty((4,))
    
    # Define a  x limit regarding maze radius + 100 pixel (arbitrary)
    #x_limit = maze_info_pixel['maze_radius_pixels']+100
    
    # Get the position halfway each vertex
    x[0] = (position_each_vertex[4,0] + position_each_vertex[5,0])/2
    y[0] = (position_each_vertex[4,1] + position_each_vertex[5,1])/2
    x[1] = (position_each_vertex[5,0] + position_each_vertex[6,0])/2
    y[1] = (position_each_vertex[5,1] + position_each_vertex[6,1])/2
    x[2] = (position_each_vertex[6,0] + position_each_vertex[7,0])/2
    y[2] = (position_each_vertex[6,1] + position_each_vertex[7,1])/2
    x[3] = (position_each_vertex[7,0] + position_each_vertex[4,0])/2
    y[3] = (position_each_vertex[7,1] + position_each_vertex[4,1])/2

    # Define the quadrants as triangles (each line is a triangle)
    quadrant1 = np.array([[centroid_coords[0], centroid_coords[1]], [x[3],y[3]], [position_each_vertex[4,0],position_each_vertex[4,1]], [x[0],y[0]]])
    quadrant2 = np.array([[centroid_coords[0], centroid_coords[1]], [x[0],y[0]], [position_each_vertex[5,0],position_each_vertex[5,1]], [x[1],y[1]]])
    quadrant3 = np.array([[centroid_coords[0], centroid_coords[1]], [x[1],y[1]], [position_each_vertex[6,0],position_each_vertex[6,1]], [x[2],y[2]]])
    quadrant4 = np.array([[centroid_coords[0], centroid_coords[1]], [x[2],y[2]], [position_each_vertex[7,0],position_each_vertex[7,1]], [x[3],y[3]]])

    # Create a dict containing all the coords for the 4 different quadrants
    quadrant_dict = dict({'quadrant1': quadrant1, 'quadrant2': quadrant2, 'quadrant3': quadrant3, 'quadrant4': quadrant4})
    
    # Create a dict with lists inside (to save as json)
    quadrant_dict_list = dict({'quadrant1': quadrant1.tolist(), 'quadrant2': quadrant2.tolist(), 'quadrant3': quadrant3.tolist(), 'quadrant4': quadrant4.tolist()})
    
    # Recreate an y (on the maze border) based on the x (radius)
    if recreate_maze is True:
        figure, axes = plt.subplots()
        # Recreate the maze
        maze_recreation_plot(axes, body_part_matrix, centroid_coords, position_each_vertex, maze_info_pixel, plot_frame=False, title='Nose', show=False, invert=False)    
        
        # Plot the lines separating the quadrants
        for i in range(4): 
            plt.plot(x[i],y[i],'.r')
            plt.plot(np.array([centroid_coords[0], x_quad[i]]), np.array([centroid_coords[1], y_quad[i]]))
        
        # Invert the yaxis
        plt.gca().invert_yaxis()
        plt.show()
    
    return quadrant_dict, quadrant_dict_list
        
        
def model_objects(dlc_data_frame, bp_names, conf_threshold, maze_info_pixel, centroid_coords, position_each_vertex, outer_obj_layer_cm=2):
    g_1_tampa = dlc_data_frame.xs(bp_names['obj_1_center'], level='bodyparts', axis=1).to_numpy()
    g_2_tampa = dlc_data_frame.xs(bp_names['obj_2_center'], level='bodyparts', axis=1).to_numpy()
    g_1_base = dlc_data_frame.xs(bp_names['obj_1_edge'], level='bodyparts', axis=1).to_numpy()
    g_2_base = dlc_data_frame.xs(bp_names['obj_2_edge'], level='bodyparts', axis=1).to_numpy()
    
    # G_1_TAMPA
    # compare the confidence for each frame 
    confidence_mask = np.where(g_1_tampa[:,2] >= conf_threshold)
    # average the (x,y) coords for only the frames that have the min confidence
    position_g_1_tampa = np.array((np.average(g_1_tampa[confidence_mask,0]), np.average(g_1_tampa[confidence_mask,1])))

    # G_2_TAMPA
    # compare the confidence for each frame 
    confidence_mask = np.where(g_2_tampa[:,2] >= conf_threshold)
    # average the (x,y) coords for only the frames that have the min confidence
    position_g_2_tampa = np.array((np.average(g_2_tampa[confidence_mask,0]), np.average(g_2_tampa[confidence_mask,1])))

    # G_1_BASE
    # compare the confidence for each frame 
    confidence_mask = np.where(g_1_base[:,2] >= conf_threshold)
    # average the (x,y) coords for only the frames that have the min confidence
    position_g_1_base = np.array((np.average(g_1_base[confidence_mask,0]), np.average(g_1_base[confidence_mask,1])))
    
    # G_2_BASE
    # compare the confidence for each frame 
    confidence_mask = np.where(g_2_base[:,2] >= conf_threshold)
    # average the (x,y) coords for only the frames that have the min confidence
    position_g_2_base = np.array((np.average(g_2_base[confidence_mask,0]), np.average(g_2_base[confidence_mask,1])))

    # Get the average x,y for each combination of base/tampa point
    av_g_1 = np.mean(np.array((position_g_1_base, position_g_1_tampa)), axis=0)
    av_g_2 = np.mean(np.array((position_g_2_base, position_g_2_tampa)), axis=0)

    # Get the linear equation regarding the object center and border OBJ 1
    x1, y1 = position_g_1_tampa
    x2, y2 = position_g_1_base
    linear_func = linear_function(x1, y1, x2, y2)
    # Evaluate the linear function at a specific x value REMEMBER to insert - for the 2
    x = position_g_1_tampa[0] + (maze_info_pixel['pixelcm_ratio'] * maze_info_pixel['obj_diameter']/2)
    
    # Get the differenc between the base and the expected border
    x_base_border_diff = (x2-x)
    # Get the new center as 'tampa' x plus the difference between base and border
    x_new_center = x1 + x_base_border_diff
    # Calc the Y value for the new center
    y_new_center = linear_func(x_new_center)
    # Update the new center coords
    av_g_1 = np.array((x_new_center, y_new_center))
    
    
    ####### Get the linear equation regarding the object center and border OBJ 1
    x2, y2 = position_g_2_tampa
    x1, y1 = position_g_2_base
    linear_func = linear_function(x1, y1, x2, y2)
    # Evaluate the linear function at a specific x value REMEMBER to insert - for the 2
    x = position_g_2_tampa[0] - (maze_info_pixel['pixelcm_ratio'] * maze_info_pixel['obj_diameter']/2)
    
    # Get the differenc between the base and the expected border
    x_base_border_diff = (x1-x)
    # Get the new center as 'tampa' x plus the difference between base and border
    x_new_center = x2 + x_base_border_diff
    # Calc the Y value for the new center
    y_new_center = linear_func(x_new_center)
    # Update the new center coords
    av_g_2 = np.array((x_new_center, y_new_center))

    
    # Define important elements regarding the objects
    # Use an empiric radius for the object - Divide it by 2 so it represents correctly the radius and not the diameter
    # emp_radius_cm = 3.5
    # emp_radius_pixel = np.array(emp_radius_cm * maze_info_pixel['pixelcm_ratio'])
    # emp_radius_pixel_outer = np.array((3.5 + outer_obj_layer_cm) * maze_info_pixel['pixelcm_ratio'])
    # maze_info_pixel['av_g_1'] = av_g_1
    # maze_info_pixel['av_g_2'] = av_g_2
    # maze_info_pixel['emp_radius_pixel'] = np.array((emp_radius_pixel*3.5, emp_radius_pixel*2.75))/2
    # maze_info_pixel['emp_radius_pixel_outer'] = np.array((emp_radius_pixel_outer*3.5, emp_radius_pixel_outer*2.75))/2
    # maze_info_pixel['angle_coeff_g_1'] = (position_g_1_tampa[1] - position_g_1_base[1])/(position_g_1_tampa[0] - position_g_1_base[0])
    # maze_info_pixel['angle_coeff_g_2'] = (position_g_2_tampa[1] - position_g_2_base[1])/(position_g_2_tampa[0] - position_g_2_base[0])
    # maze_info_pixel['angle_g_1'] = math.degrees(math.atan(maze_info_pixel['angle_coeff_g_1']))
    # maze_info_pixel['angle_g_2'] = math.degrees(math.atan(maze_info_pixel['angle_coeff_g_2']))
    

    emp_radius_cm = maze_info_pixel['obj_diameter']/2
    emp_radius_pixel = np.array(emp_radius_cm * maze_info_pixel['pixelcm_ratio'])
    emp_radius_pixel_outer = np.array((emp_radius_cm + outer_obj_layer_cm) * maze_info_pixel['pixelcm_ratio'])
    maze_info_pixel['av_g_1'] = av_g_1
    maze_info_pixel['av_g_2'] = av_g_2
    maze_info_pixel['emp_radius_pixel'] = np.array((emp_radius_pixel*3, emp_radius_pixel*2.75))/2
    maze_info_pixel['emp_radius_pixel_outer'] = np.array((emp_radius_pixel_outer*3, emp_radius_pixel_outer*2.75))/2
    maze_info_pixel['angle_coeff_g_1'] = (position_g_1_tampa[1] - position_g_1_base[1])/(position_g_1_tampa[0] - position_g_1_base[0])
    maze_info_pixel['angle_coeff_g_2'] = (position_g_2_tampa[1] - position_g_2_base[1])/(position_g_2_tampa[0] - position_g_2_base[0])
    maze_info_pixel['angle_g_1'] = math.degrees(math.atan(maze_info_pixel['angle_coeff_g_1']))
    maze_info_pixel['angle_g_2'] = math.degrees(math.atan(maze_info_pixel['angle_coeff_g_2']))
    
    # Convert arrays to pixels
    maze_info_pixel_list = maze_info_pixel
    maze_info_pixel_list['av_g_1'] = av_g_1.tolist()
    maze_info_pixel_list['av_g_2'] = av_g_2.tolist()
    maze_info_pixel_list['emp_radius_pixel'] = maze_info_pixel['emp_radius_pixel'].tolist()
    maze_info_pixel_list['emp_radius_pixel_outer'] = maze_info_pixel['emp_radius_pixel_outer'].tolist()

    fig, axes = plt.subplots()
    maze_recreation_plot_OLR(axes, centroid_coords, position_each_vertex, maze_info_pixel)
    
    return maze_info_pixel, maze_info_pixel_list
    