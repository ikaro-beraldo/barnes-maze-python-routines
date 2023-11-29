# Set of functions for data analysis such as used strategy, animal speed, time spent on a certain part of the maze

from minors_functions import *
from matplotlib import pyplot as plt
import numpy as np
import sympy as sp
from color_functions import *
from plot_functions import maze_recreation_plot_OLR

# Extract the animal position relative to the maze holes
def get_bp_position_on_maze(body_part_matrix, position_each_hole, maze_info_pixel, centroid_coords, fps=30):
    
    x = body_part_matrix[:,0]  # x coords
    y = body_part_matrix[:,1]  # y coords
    hole_radius = maze_info_pixel["hole_radius_pixels"]
    maze_radius = maze_info_pixel["maze_radius_pixels"]
    
    # Create a body part position vector (coded as 1 to 12 for holes, 0 for the maze and -1 when outside the maze)
    bp_pos_on_maze = np.zeros((len(x),))
    
    # Loop for each frame
    for i in range(len(x)):
        # Loop for each hole
        for ii in range(np.shape(position_each_hole)[0]):
            # Check if the body part is inside the specific hole
            if isInside(position_each_hole[ii,0], position_each_hole[ii,1], hole_radius, x[i], y[i]): 
                bp_pos_on_maze[i] = ii + 1  # Insert the index regarding the role (+1, since python indexing is quite differente from what I've learned in MATLAB)

    ## Check if the body part is outside the maze (ignore the parts which are already on holes)
    on_maze = np.where(bp_pos_on_maze == 0)
    for i in np.transpose(on_maze):
        if not(isInside(centroid_coords[0], centroid_coords[1], maze_radius, x[i], y[i])): # If the body part is inside the maze
            bp_pos_on_maze[i] = -1  # insert the code 1
    
    return bp_pos_on_maze

def get_sequence_and_duration(arr):
    # Find the indices where the array changes
    change_indices = np.where(np.diff(arr) != 0)[0] + 1

    # Split the array at the change indices to get consecutive repetitions
    sequences = np.split(arr, change_indices)

    # Find unique elements and their counts in each sequence
    sequences_info = [(np.unique(seq), np.diff(np.where(np.concatenate(([True], seq[:-1] != seq[1:], [True])))[0])) for seq in sequences]

    unique_elements = np.zeros((len(sequences_info),))  # Values array
    counts = np.zeros((len(sequences_info),))           # Counts array
    # Unpack the results
    i = 0;
    for elem in sequences_info:
        unique_elements[i], counts[i] = elem[0], elem[1]
        i += 1

    return np.array(unique_elements), np.array(counts)

# Extract the number of primary errors on Barnes Maze (default target hole = 1)
def get_p_s_errors(bp_pos_on_maze,target=1, exp_min_time=1, fps=30):
    # Get the entries and exits from holes
    #entrie_exit = np.array(np.where(np.diff(bp_pos_on_maze) != 0))
    
    # Find unique hole exploration and their counts
    unique_elements, counts = get_sequence_and_duration(bp_pos_on_maze)
    
    # Compute the total number of errors
    errors = (unique_elements != target) & (unique_elements != 0) & (unique_elements != -1)
    
    # Compute the total number of hole explorations (consider target exploration)
    exploration = (unique_elements != 0) & (unique_elements != -1)
    
    # Compute the total number of errors
    #errors = (bp_pos_on_maze[np.array(entrie_exit)+1] != target) & (bp_pos_on_maze[np.array(entrie_exit)+1] != 0) & (bp_pos_on_maze[np.array(entrie_exit)+1] != -1)
    
    # Get the number of primary errors
    p_errors = len(np.unique(unique_elements[errors]))
    
    # Separate only the hole explorations in which the exploration time is >= exp_min_time (in sec)
    s_errors = len(np.where(counts[errors]/fps >= exp_min_time)[0]) - p_errors
    # Check if s_errors will be a negative number
    if s_errors < 0:
        s_errors = 0
        
    # Get the number of secundary
    #s_errors = len(bp_pos_on_maze[np.array(entrie_exit[errors])+1]) - p_errors  
    
    # Create an array with the errors in order of appearence
    h_errors = np.zeros((len(unique_elements[errors]),))    # Pre-allocate it
    for i in np.unique(unique_elements[errors]):            # Get all the holes wrongly searched
        # Get the first primary error holes
        h_errors[np.where(unique_elements[errors] == i)[0][0]] = i
        
        # Get the specific hole errors index (which surpass the minimum exploration time)
        idx_above_min_time = unique_elements[errors][np.where(counts[errors] / fps >= exp_min_time)] == i
        # Add the secundary errors to this array
        h_errors[np.where(counts[errors] / fps >= exp_min_time)[0][idx_above_min_time]] = i
    
    # Delete errors which are not primary neither passed the exp_min_time
    h_errors = np.delete(h_errors, np.where(h_errors == 0))
    
    # Create an array with the exploration in order of appearence
    h_exp = np.zeros((len(unique_elements[exploration]),))    # Pre-allocate it
    for i in np.unique(unique_elements[exploration]):            # Get all the holes wrongly searched
        # Get the first primary error holes
        h_exp[np.where(unique_elements[exploration] == i)[0][0]] = i
        
        # Get the specific hole exploration index (which surpass the minimum exploration time)
        idx_above_min_time = unique_elements[exploration][np.where(counts[exploration] / fps >= exp_min_time)] == i
        # Add the secundary exploration to this array
        h_exp[np.where(counts[exploration] / fps >= exp_min_time)[0][idx_above_min_time]] = i
    
    # Delete exploration which are not primary neither passed the exp_min_time
    h_exp = np.delete(h_exp, np.where(h_exp == 0))
    
    # Holes regarding the errors
    #h_errors = bp_pos_on_maze[np.array(entrie_exit[errors])+1]
    
    # Indices regarding the errors
    #i_errors = np.array(entrie_exit[errors])+1
    
        
    return p_errors, s_errors, h_errors, h_exp

def get_distance(body_part_matrix, maze_info_pixel):
    # Euclidean distance
    euc_distance_frame = np.array(np.sqrt(np.diff(body_part_matrix[:,0])**2 + np.diff(body_part_matrix[:,1])**2))
    # Convert to cm
    euc_distance_frame = euc_distance_frame/maze_info_pixel["pixelcm_ratio"]
    # Add a 0 for the first frame
    euc_distance_frame = np.append([0], euc_distance_frame, axis=0)
    # Get the total distance
    total_distance = np.sum(euc_distance_frame)

        
    return euc_distance_frame, total_distance

# Function to extract the animals instantaneous speed
def get_inst_speed(body_part_matrix, maze_info_pixel, time_win=1, fps=30):
    # Get the distance frame by frame (only the first output)
    euc_distance_frame = get_distance(body_part_matrix, maze_info_pixel)[0]
    # Time window in sec
    time_win_sec = time_win/fps
    # Calculate speed
    n_bins = np.int64(np.floor(euc_distance_frame.shape[0]/time_win))
    inst_speed = np.zeros((n_bins,))
    inst_speed_entire = np.zeros((len(euc_distance_frame),))
    
  
    for i in range(len(inst_speed)):
        # Instantaneous speed based on the time window
        a = time_win*i             # First element
        b = time_win*(i+1) - 1     # Last element
        inst_speed[i] = np.sum(euc_distance_frame[np.arange(a,b+1,1)])/time_win_sec    # isntantaneous speed
        inst_speed_entire[np.arange(a,b+1,1)] = np.sum(euc_distance_frame[np.arange(a,b+1,1)])/time_win_sec       # Instaneous speed, but filling every single frame value (even if it is repeated, important for plot)
           
    # Mean speed
    av_speed = np.mean(inst_speed)
        
    return inst_speed, inst_speed_entire, av_speed
        
    
# Define the strategy used by the animal
def get_the_strategy(bp_pos_on_maze, target=1, exp_min_time=1, fps=30):
    # First get the primary and secundary errors, including the hole numbers
    p_errors, s_errors, h_errors, h_exp = get_p_s_errors(bp_pos_on_maze, target, exp_min_time=exp_min_time, fps=fps)[0:4]
    
    # If it doesn't fall to any of those conditions, classify the strategy as random
    strategy = 'random'
    
    # Direct/Spatial strategy
    if p_errors + s_errors < 3:
        # Check if holes were whithin 2 from target (only holes with difference 1, 3, 11 and 10)
        if not(((np.abs(h_errors-target) > 2) & (np.abs(h_errors-target) < 10)).any()):
            strategy = 'spatial'
        elif p_errors + s_errors <= 1: # One error far from the target or 0 errors
            strategy = 'spatial'
        else:
            strategy = 'random'
    
    # (serial/random)
    else:
        # # Add the target role to the end of h_erros if it is serial
        # h_errors = np.append([[h_errors]],[[target]])
        # # Check if at least 75% of the exploration was serial (
        # serial_mask = ((np.abs(np.diff(h_errors)) == 1) | (np.abs(np.diff(h_errors)) == 0) | (np.abs(np.diff(h_errors)) == 11).all())
        
        # Check if at least 75% of the exploration was serial (
        serial_mask = (np.abs(np.diff(h_exp)) == 1) | (np.abs(np.diff(h_exp)) == 0) | (np.abs(np.diff(h_exp)) == 11)
                
        # If 75% of searchs were serial
        if np.array(np.where(serial_mask == True)).shape[1] >= len(h_errors)*0.75:
            strategy = 'serial'
        else:
            strategy = 'random'
    
    return strategy


# Function to get the trial latency
def get_trial_latency(body_part_matrix, fps=30):
    # get the number of frames
    n_frames = np.shape(body_part_matrix)[0]
    # latency
    latency = n_frames/fps
    
    return latency

# Function to get the trial latency regarding the time when the animal finds the scape hole
def get_trial_latency_to_find_scape(body_part_matrix_nose, body_part_matrix_body_centre, bp_pos_on_maze, bp_pos_on_maze_filtered, time_thresh=3, fps=30, scape_hole=1):
    # time_thresh is the time in seconds that the animal must explore the scape_hole 
    
    # Get the frames which the animal explored the escape hole
    a = np.zeros(len(bp_pos_on_maze),)
    a[bp_pos_on_maze==scape_hole] = 1
    # Diff of index 
    dif = np.diff(a)
    
    # Beginning of exploration
    b = np.where(dif==1)[0]+1
    # End of exploration
    e = np.where(dif==-1)[0]
    
    # if in the first frame the animal is exploring the scape hole (almost impossible... but who knows)
    if a[0] == 1:
        # Beginning of exploration
        b = np.hstack((0, b))
    # if in the last frame the animal is exploring the scape hole
    if a[-1] == 1:
        e = np.hstack((e, len(bp_pos_on_maze)))
    
    # Get which exploration event lasted more than the threshold
    exp_bool = (e-b)/fps >= time_thresh
    # In case some of the explorations lasted more than the threshold
    if np.any(exp_bool):
        first_exp = np.where(exp_bool==True)[0]
        # The latency is the frame which the animal started to explore + the time_threshold
        latency_frame = int(b[first_exp][0] + time_thresh*fps)
        latency = latency_frame/fps
    else:   # If none of the explorations lasted more than the threshold, compute the scape latency
        latency = get_trial_latency(body_part_matrix_nose, fps=fps)
        latency_frame = int(latency*fps)
        
    # Fix the previous vectors (bp_matrix, bp_pos_on_maze, etc)
    body_part_matrix_nose = body_part_matrix_nose[:latency_frame+1]
    body_part_matrix_body_centre = body_part_matrix_body_centre[:latency_frame+1]
    bp_pos_on_maze = bp_pos_on_maze[:latency_frame+1]
    bp_pos_on_maze_filtered = bp_pos_on_maze_filtered[:latency_frame+1]
    
    return latency, body_part_matrix_nose, body_part_matrix_body_centre, bp_pos_on_maze, bp_pos_on_maze_filtered


# Extract the animal position relative to the maze quadrants
def get_bp_position_on_quadrant(body_part_matrix, quadrant_dict, fps=30):
    
    x = body_part_matrix[:,0]  # x coords
    y = body_part_matrix[:,1]  # y coords
    
    
    # Create a body part position vector (coded as 1 to 4 for each one of the quadrants)
    bp_pos_on_quadrant = np.zeros((len(x),))
    
    # Loop for each frame
    for i in range(len(x)):
        # Loop for each quadrant
        compare = np.zeros((4,))
        for ii in range(len(quadrant_dict)):
            # Check if the body part is positioned on the specific
            triangle = quadrant_dict['quadrant'+str(ii+1)]            
            
            # Get the diference between areas -> inference for animal distance
            compare[ii] = isInside_triangle_dif(triangle, x[i], y[i])
        
        # Get the index of the nearest quadrant (considering the animal's bp)       
        bp_pos_on_quadrant[i] = np.argmin(compare) + 1 # Insert the index regarding the role (+1, since python indexing is quite differente from what I've learned in MATLAB)

    return bp_pos_on_quadrant

# Extract the animal position relative to the maze quadrants
def get_bp_position_on_quadrant_OLR(body_part_matrix, quadrant_dict, fps=30):
    
    xy = body_part_matrix[:,0:2]  # x,y coords       
    # Create a body part position vector (coded as 1 to 4 for each one of the quadrants)
    bp_pos_on_quadrant = np.zeros(xy.shape[0],)
    
    # Loop for each quadrant
    for ii in range(len(quadrant_dict)):
        # Check if the body part is positioned on the specific
        rectangle = quadrant_dict['quadrant'+str(ii+1)]      
        # Check whether each point is inside the quadrant currently being analyzed
        results_bool = are_points_inside_polygon(rectangle, xy)        
        # Insert the index regarding the animal's body part on quadrant      
        bp_pos_on_quadrant[results_bool] = ii + 1 # Insert the index regarding the role (+1, since python indexing is quite differente from what I've learned in MATLAB)

    return bp_pos_on_quadrant

# Get the time spent on each quadrant and calculate the ratio time on the target quadrant / time on the other quadrants
def get_time_on_target_quadrant(bp_pos_on_quadrant, target=1, fps=30, prob=False):
    # Create an zeros array
    time_on_each_quadrant = np.zeros(4,)
    # Get the time on each quadrant
    time_on_each_quadrant[0] = np.count_nonzero(bp_pos_on_quadrant == 1)/fps
    time_on_each_quadrant[1] = np.count_nonzero(bp_pos_on_quadrant == 2)/fps
    time_on_each_quadrant[2] = np.count_nonzero(bp_pos_on_quadrant == 3)/fps
    time_on_each_quadrant[3] = np.count_nonzero(bp_pos_on_quadrant == 4)/fps
    
    # Calculate the ratio target/others
    ratio_target_others = time_on_each_quadrant[target-1]/np.sum(time_on_each_quadrant)
    
    # Just pre-allocate
    ratio_target_minute = np.zeros((2,))
    
    # Also calculate the time by minute of trial  
    if prob is True:
        a = bp_pos_on_quadrant[0:60*fps-1]
        ratio_target_minute[0] = np.count_nonzero(a == target) / (60*fps)
        
        b = bp_pos_on_quadrant[60*fps:120*fps]
        ratio_target_minute[1] = np.count_nonzero(b == target) / (60*fps)
    
    return ratio_target_others, time_on_each_quadrant, ratio_target_minute

def get_head_angle(body_part_matrix_nose,body_part_matrix_head):
    # Create a dict
    head_angle = dict({'slope': '', 'degree': ''})
    
    nose = body_part_matrix_nose[:,0:2] # Get the nose x and y coords
    head = body_part_matrix_head[:,0:2] # Get the head x and y coords
    
    # Compute the head angle as the line slope or degree
    head_angle['slope'] = (nose[:,1] - head[:,1])/(nose[:,0] - head[:,0])
    head_angle['degree'] = np.degrees(np.arctan(head_angle['slope']))
    
    return head_angle

# Extract the animal position relative to the maze holes
def get_bp_position_on_maze_OLR(body_part_matrix_nose, maze_info_pixel, centroid_coords, position_each_vertex, fps=30):
    
    # Nose coords
    x = body_part_matrix_nose[:,0]  # x coords
    y = body_part_matrix_nose[:,1]  # y coords
       
    # Create a body part position vector (coded as 1 to 12 for holes, 0 for the maze and -1 when outside the maze)
    bp_pos_on_maze = np.zeros((len(x),))
        
    # Outer Ellipse Obj 1
    # Check if the body part is inside the specific obj ellipse
    mask = is_point_inside_ellipse(x, y, maze_info_pixel['emp_radius_pixel_outer'][0], maze_info_pixel['emp_radius_pixel_outer'][1], maze_info_pixel['av_g_1'][0], maze_info_pixel['av_g_1'][1], sp.rad(maze_info_pixel['angle_g_1']))
    bp_pos_on_maze[mask] = 1.5  # Insert the index regarding the obj (+1)

    # Inner Ellipse Obj 1
    mask = is_point_inside_ellipse(x, y, maze_info_pixel['emp_radius_pixel'][0], maze_info_pixel['emp_radius_pixel'][1], maze_info_pixel['av_g_1'][0], maze_info_pixel['av_g_1'][1], sp.rad(maze_info_pixel['angle_g_1']))
    bp_pos_on_maze[mask] = 1  # Insert the index regarding the obj (+1)

    # Outer Ellipse Obj 2
    # Check if the body part is inside the specific obj ellipse
    mask = is_point_inside_ellipse(x, y, maze_info_pixel['emp_radius_pixel_outer'][0], maze_info_pixel['emp_radius_pixel_outer'][1], maze_info_pixel['av_g_2'][0], maze_info_pixel['av_g_2'][1], sp.rad(maze_info_pixel['angle_g_2']))
    bp_pos_on_maze[mask] = 2.5  # Insert the index regarding the obj (+1)

    # Inner Ellipse Obj 2
    mask = is_point_inside_ellipse(x, y, maze_info_pixel['emp_radius_pixel'][0], maze_info_pixel['emp_radius_pixel'][1], maze_info_pixel['av_g_2'][0], maze_info_pixel['av_g_2'][1], sp.rad(maze_info_pixel['angle_g_2']))
    bp_pos_on_maze[mask] = 2  # Insert the index regarding the obj (+1)
            
        
    # video_file = "F:\\Barnes Maze - Mestrad\\OLR-RAFA\\T5 EvocacaoDLC_resnet50_OLRJun15shuffle2_1000000_labeled.mp4"
    
    # for i in range(bp_pos_on_maze.size):
    #     idx = i
    #     classification = bp_pos_on_maze[i]
    #     fig, axes = plt.subplots()
    #     maze_recreation_plot_OLR(axes, centroid_coords, position_each_vertex, maze_info_pixel, video_file=video_file, video_frame=idx, show=False, invert=False)
    #     plot_ellipse(axes, maze_info_pixel['emp_radius_pixel_outer'][0], maze_info_pixel['emp_radius_pixel_outer'][1], maze_info_pixel['av_g_1'][0], maze_info_pixel['av_g_1'][1], sp.rad(maze_info_pixel['angle_g_1']))
    #     axes.plot(x[i],y[i],'.', markersize=5)
    #     axes.invert_yaxis()
    #     print(classification)
    #     plt.show()
    
    
    return bp_pos_on_maze

def plot_ellipse(axes, a, b, h, k, A):
    # Convert parameters to regular Python floats
    a, b, h, k, A = float(a), float(b), float(h), float(k), float(A)

    # Create a NumPy array of angles for plotting the ellipse
    theta = np.linspace(0, 2 * np.pi, 100)

    # Parametric equations of the ellipse
    x_ellipse = h + a * np.cos(theta) * np.cos(A) - b * np.sin(theta) * np.sin(A)
    y_ellipse = k + a * np.cos(theta) * np.sin(A) + b * np.sin(theta) * np.cos(A)

    # Plot the ellipse and the line
    axes.plot(x_ellipse, y_ellipse, label='Ellipse')
    
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.title('Ellipse and Line Intersection')
    # plt.legend()
    # plt.grid(True)
    # plt.axis('equal')
    # plt.show()

def plot_ellipse_and_line(axes, a, b, h, k, A, line_a, line_b, intersections):
    # Convert parameters to regular Python floats
    a, b, h, k, A = float(a), float(b), float(h), float(k), float(A)

    # Create a NumPy array of angles for plotting the ellipse
    theta = np.linspace(0, 2 * np.pi, 100)

    # Parametric equations of the ellipse
    x_ellipse = h + a * np.cos(theta) * np.cos(A) - b * np.sin(theta) * np.sin(A)
    y_ellipse = k + a * np.cos(theta) * np.sin(A) + b * np.sin(theta) * np.cos(A)

    # Parametric equations of the line
    x_line = np.linspace(min(x_ellipse), max(x_ellipse), 100)
    y_line = line_a * x_line + line_b

    # Plot the ellipse and the line
    axes.plot(x_ellipse, y_ellipse, label='Ellipse')
    axes.plot(x_line, y_line, label='Line')

    # Plot the intersection points
    if intersections.size > 0:
        axes.scatter(intersections[:, 0], intersections[:, 1], color='red', label='Intersections')

    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.title('Ellipse and Line Intersection')
    # plt.legend()
    # plt.grid(True)
    # plt.axis('equal')
    #plt.show()
    
# Function to define which object the animal is exploring throughout time
# Get the HEAD y to define b for a linear equation
def get_obj_exploration(bp_pos_on_maze,head_angle, maze_info_pixel, body_part_matrix_nose, body_part_matrix_head, centroid_coords, position_each_vertex, fps=30):
    # Iniate the final obj_exploration variable
    obj_exploration = np.zeros((len(bp_pos_on_maze),))
    
    # If the animal nose is already on the inner ellipses (on the object) consider it as exploration
    obj_exploration[bp_pos_on_maze==1] = 1
    obj_exploration[bp_pos_on_maze==2] = 2
    
    # Check for the exploration on the outer area
    # Find when the animal nose is in the object outer area
    # Obj 1
    mask_1 = np.array(np.where(bp_pos_on_maze == 1.5))
    
    video_file = "F:\\Barnes Maze - Mestrad\\OLR-RAFA\\T6_Evocacao_1DLC_resnet50_OLRJun15shuffle2_1000000_labeled.mp4"
    
    for i in range(mask_1.size):
        idx = mask_1[0,i]
        points = [tuple(body_part_matrix_head[idx,0:2]), tuple(body_part_matrix_nose[idx,0:2])]
        m, b = solve_m_c_linear_equation(points)
        intersections = find_line_ellipse_intersections(maze_info_pixel['emp_radius_pixel'][0], maze_info_pixel['emp_radius_pixel'][1], maze_info_pixel['av_g_1'][0], maze_info_pixel['av_g_1'][1], np.radians(maze_info_pixel['angle_g_1']), m, b)
        # fig, axes = plt.subplots()
        # maze_recreation_plot_OLR(axes, centroid_coords, position_each_vertex, maze_info_pixel, video_file=video_file, video_frame=idx, show=False, plot_frame=False)
        # plot_ellipse_and_line(axes, maze_info_pixel['emp_radius_pixel_outer'][0], maze_info_pixel['emp_radius_pixel_outer'][1], maze_info_pixel['av_g_1'][0], maze_info_pixel['av_g_1'][1], sp.rad(maze_info_pixel['angle_g_1']), m, b, intersections)
        # x = np.array([body_part_matrix_nose[idx,0], body_part_matrix_head[idx,0]])
        # y = np.array([body_part_matrix_nose[idx,1], body_part_matrix_head[idx,1]])
        # axes.plot(x,y)
        # plt.show()
        
        # Check if any intersection was found 
        if intersections.size > 0:
            obj_exploration[idx] = 1
            
            
    # Obj 2
    mask_2 = np.array(np.where(bp_pos_on_maze == 2.5))
    
    video_file = "F:\\Barnes Maze - Mestrad\\OLR-RAFA\\T6_Evocacao_1DLC_resnet50_OLRJun15shuffle2_1000000_labeled.mp4"
    
    for i in range(mask_2.size):
        idx = mask_2[0,i]
        points = [tuple(body_part_matrix_head[idx,0:2]), tuple(body_part_matrix_nose[idx,0:2])]
        m, b = solve_m_c_linear_equation(points)
        intersections = find_line_ellipse_intersections(maze_info_pixel['emp_radius_pixel'][0], maze_info_pixel['emp_radius_pixel'][1], maze_info_pixel['av_g_2'][0], maze_info_pixel['av_g_2'][1], np.radians(maze_info_pixel['angle_g_2']), m, b)
        # fig, axes = plt.subplots()
        # maze_recreation_plot_OLR(axes, centroid_coords, position_each_vertex, maze_info_pixel, video_file=video_file, video_frame=idx, show=False, plot_frame=False)
        # plot_ellipse_and_line(axes, maze_info_pixel['emp_radius_pixel_outer'][0], maze_info_pixel['emp_radius_pixel_outer'][1], maze_info_pixel['av_g_2'][0], maze_info_pixel['av_g_2'][1], sp.rad(maze_info_pixel['angle_g_2']), m, b, intersections)
        # x = np.array([body_part_matrix_nose[idx,0], body_part_matrix_head[idx,0]])
        # y = np.array([body_part_matrix_nose[idx,1], body_part_matrix_head[idx,1]])
        # axes.plot(x,y)
        # plt.show()
        
        # Check if any intersection was found 
        if intersections.size > 0:
            obj_exploration[idx] = 2
            
    # Compute important parameters 
    obj_exp_parameters = dict()
    obj_exp_parameters['time_obj_1'] = np.array(np.where(obj_exploration==1)).shape[1]/fps
    obj_exp_parameters['time_obj_2'] = np.array(np.where(obj_exploration==2)).shape[1]/fps
    obj_exp_parameters['time_objs'] =  obj_exp_parameters['time_obj_1'] +  obj_exp_parameters['time_obj_2']
    obj_exp_parameters['ratio_1_2'] =  obj_exp_parameters['time_obj_1'] / obj_exp_parameters['time_obj_2']
    obj_exp_parameters['ratio_2_1'] =  obj_exp_parameters['time_obj_2'] / obj_exp_parameters['time_obj_1']
    obj_exp_parameters['ratio_1_total'] =  obj_exp_parameters['time_obj_1'] / obj_exp_parameters['time_objs']
    obj_exp_parameters['ratio_2_total'] =  obj_exp_parameters['time_obj_2'] / obj_exp_parameters['time_objs']

    return obj_exploration, obj_exp_parameters
    