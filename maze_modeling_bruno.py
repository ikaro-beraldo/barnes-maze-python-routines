# Set of functions regarding the Barnes Maze (pixel/cm calc; centroid inference; Maze recreation)

import numpy as np
from matplotlib import pyplot as plt
from plot_functions import *
import statistics as st
import math

def get_hole_positions(dlc_data_frame, filename, prev_hole_position=None):
    min_confidence = 0.99    # Minimum confidence threshold
    # Create an empty matrix for the holes coordinates
    position_each_hole = np.zeros((12,2))
    # Loop for each one the 12 holes
    for i in range(12):
        # Convert the hole x,y and confidence values to numpy array
        hole = dlc_data_frame.xs('h_'+str(i+1), level='bodyparts', axis=1).to_numpy()
        
        # Check the distribution of the confidence values.
        # If most (majority_lim) values of confidence are below a certain threshold 
        # (min_confidence), we consider the median of the data to determine 
        # the appropriate position for the point
        conf = hole[:,2]
        majority_lim = 0.75 # [%]
        if (np.count_nonzero(conf < min_confidence) > majority_lim*len(conf)):
            # Calculate the median of the coordinates
            # (Ikaro my friend, I agree that the mode is the most appropriate metric. However, 
            # since the mode is the number that occurs most frequently, 
            # we may have a problem if the coordinates are all different from each other.
            # In this case, there will be no repetition at all. Implementationally-wise,
            # maybe the median will be simpler and still do the job. What do you think?)
            position_each_hole[i,:] = np.array((st.median(hole[:,0]), st.median(hole[:,1])))
        else:
            # compare the confidence for each frame 
            confidence_mask = np.where(hole[:,2] >= min_confidence)
            # average the (x,y) coords for only the frames that have the min confidence
            position_each_hole[i,:] = np.array((np.average(hole[confidence_mask,0]), np.average(hole[confidence_mask,1])))
        
    # Manually correct the hole(s) in some videos
    if (filename.find('B01_13_G3_D4_T2') != -1):
        position_each_hole = np.insert(position_each_hole, [8], [497.741, 160.076], axis=0)
        position_each_hole = position_each_hole[0:len(position_each_hole)-1]
    elif (filename.find('B01_10_G2_D16_T1') != -1):
        position_each_hole[10] = [822.100, 113.900]
    elif (filename.find('B01_14_G3_D3_T3') != -1):
        position_each_hole[9] = [629.862, 79.3251]
    elif (filename.find('B01_14_G3_D4_T1') != -1):
        position_each_hole = np.insert(position_each_hole, [8], [497.669, 157.655], axis=0)
        position_each_hole = position_each_hole[0:len(position_each_hole)-1]
    elif (filename.find('B01_14_G3_D4_T3') != -1):
        position_each_hole = np.insert(position_each_hole, [9], [618.858, 86.8223], axis=0)
        position_each_hole = position_each_hole[0:len(position_each_hole)-1]
        
        
    # Check if all holes got valid coords values, if not, get a prev_hole_position
    if np.sum(np.isnan(position_each_hole)) != 0:
        r,c = np.where(np.isnan(position_each_hole)==True)  # Get the indices where it is NaN
        for i,j in zip(r,c): # Loop for each NaN value
            position_each_hole[i,j] = prev_hole_position[i,j]
        
    
    return position_each_hole

# Calculate the Maze centroid based on the 12 holes coordinates
def centroid_inference(hole_coords):
    x = np.sum(hole_coords[:,0])/len(hole_coords[:,0])
    y = np.sum(hole_coords[:,1])/len(hole_coords[:,1])
    
    return np.array((x,y))

# Recreate the maze area accordingly to real and pixel measurements
def maze_recreation(hole_coords, distance_from_edge, hole_radius, maze_radius):
    # Get the distance between the center of holes 1 to 7
    real_dist_holes = hole_radius*2 + 14 # Real distance in cm
    #print('real dist holes =' + str(real_dist_holes))
    pixel_dist_holes = np.empty((11,1))
    for i in range(11):
        pixel_dist_holes[i] = np.sqrt((hole_coords[i,0]-hole_coords[i+1,0])**2 + (hole_coords[i,1]-hole_coords[i+1,1])**2) # Pixel distance
        #print(pixel_dist_holes[i])
    
    pixel_dist_holes = np.mean(pixel_dist_holes)
    #print('final:' + str(pixel_dist_holes))
    
    # Create variables with important parameters
    pixelcm_ratio = pixel_dist_holes/real_dist_holes
    maze_radius_pixels =  maze_radius*pixelcm_ratio
    dist_from_edge_pixels = distance_from_edge*pixelcm_ratio
    hole_radius_pixels = hole_radius*pixelcm_ratio
    
    # Produce a dict with the maze elements PIXEL length values
    maze_info_pixels = dict({'pixelcm_ratio': pixelcm_ratio, 'maze_radius_pixels': maze_radius_pixels,  'dist_from_edge_pixels': dist_from_edge_pixels, 'hole_radius_pixels': hole_radius_pixels})
    
    return maze_info_pixels


# Function to define the maze quadrants
def maze_quadrants(maze_info_pixel, body_part_matrix, centroid_coords, position_each_hole, plot_frame=False, title='Nose', show=True, recreate_maze=False):
    # Get the points between the holes 2/3, 5/6, 8/9, 11/12
    ref_holes = np.transpose(np.array([2, 5, 8, 11]))
    
    # Pre-allocate the x and y vectors
    x = np.empty((4,1))
    y = np.empty((4,1))
    
    # Define a x limit regarding maze radius + 100 pixel (arbitrary)
    x_limit = maze_info_pixel['maze_radius_pixels']+300
    
    for i in range(np.shape(x)[0]):
        idx = ref_holes[i] # hole idx
        x[i] = (position_each_hole[idx-1,0] + position_each_hole[idx,0])/2
        y[i] = (position_each_hole[idx-1,1] + position_each_hole[idx,1])/2
    
    # Variable to get angular coefficient 
    a = np.empty((4,)) 
    theta = np.empty((4,)) 
 
    # Create a function for each divisory line
    for i in range(4):
        # Get the angle of the line from the centroid to the points x and y
        theta[i] = math.atan2((y[i]-centroid_coords[1]), (x[i]-centroid_coords[0]))
        
    x_quad = np.empty((4,))
    y_quad = np.empty((4,))
    
    # Get the border x and y coordinates for the line quadrants
    for i in range(4):
        x_quad[i], y_quad[i] = math.cos(theta[i])*x_limit + centroid_coords[0], math.sin(theta[i])*x_limit + centroid_coords[1]

    # Define the quadrants as triangles (each line is a triangle)
    quadrant1 = np.array([[centroid_coords[0], centroid_coords[1]], [x_quad[3],y_quad[3]], [x_quad[0],y_quad[0]]])
    quadrant2 = np.array([[centroid_coords[0], centroid_coords[1]], [x_quad[0],y_quad[0]], [x_quad[1],y_quad[1]]])
    quadrant3 = np.array([[centroid_coords[0], centroid_coords[1]], [x_quad[1],y_quad[1]], [x_quad[2],y_quad[2]]])
    quadrant4 = np.array([[centroid_coords[0], centroid_coords[1]], [x_quad[2],y_quad[2]], [x_quad[3],y_quad[3]]])

    #######################
    #plt.plot(position_each_hole[:,0],-position_each_hole[:,1])
    #plt.plot(centroid_coords[0],-centroid_coords[1],'.')
    #plt.plot(x,-y,'.')
    #plt.plot(x_quad, -y_quad, '.')
    ###########################
    
    # Create a dict containing all the coords for the 4 different quadrants
    quadrant_dict = dict({'quadrant1': quadrant1, 'quadrant2': quadrant2, 'quadrant3': quadrant3, 'quadrant4': quadrant4})
    
    # Create a dict with lists inside (to save as json)
    quadrant_dict_list = dict({'quadrant1': quadrant1.tolist(), 'quadrant2': quadrant2.tolist(), 'quadrant3': quadrant3.tolist(), 'quadrant4': quadrant4.tolist()})
    
    # Recreate an y (on the maze border) based on the x (radius)
    if recreate_maze is True:
        figure, axes = plt.subplots()
        # Recreate the maze
        maze_recreation_plot(axes, body_part_matrix, centroid_coords, position_each_hole, maze_info_pixel, plot_frame=False, title='Nose', show=False, invert=False)    
        
        # Plot the lines separating the quadrants
        for i in range(4): 
            plt.plot(x[i],y[i],'.r')
            plt.plot(np.array([centroid_coords[0], x_quad[i]]), np.array([centroid_coords[1], y_quad[i]]))
        
        # Invert the yaxis
        plt.gca().invert_yaxis()
        plt.show()
    
    return quadrant_dict, quadrant_dict_list
        
        
    