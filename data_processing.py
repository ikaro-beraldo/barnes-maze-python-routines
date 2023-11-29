# Set of functions to process the DLC outputs
import numpy as np
from scipy import stats as sts
from matplotlib import pyplot as plt
from minors_functions import *

def fix_frames_confidence(body_part_matrix,conf_threshold):
    
    # Get a confidence mask based on the confidence threshold
    confidence_mask = body_part_matrix[:,2] >= conf_threshold
    x_fixed = body_part_matrix[:,0] # X coords
    y_fixed = body_part_matrix[:,1] # Y coords
   
    exc_frames = np.transpose(np.asarray(np.where(confidence_mask == False)))   # Coords excluded (<0.95)
    acc_frames = np.transpose(np.asarray(np.where(confidence_mask == True)))  # Coords accepted (>0.95)

    first_acc = acc_frames[0]  # First accepted frame
    last_acc = acc_frames[-1]  # Last accepted frame

    # Loop for each excluded frame
    for i in range(len(exc_frames)):
        if exc_frames[i] < first_acc: # Frames excluded before the first one accepted
            x_fixed[exc_frames[i]] = body_part_matrix[first_acc,0]
            y_fixed[exc_frames[i]] = body_part_matrix[first_acc,1]
        elif exc_frames[i] > last_acc: # Frames excluded after the last one accepted
            x_fixed[exc_frames[i]] = body_part_matrix[last_acc,0]
            y_fixed[exc_frames[i]] = body_part_matrix[last_acc,1]
        else: # Frames which are inserted between accepted frames
            x_fixed[exc_frames[i]] = body_part_matrix[exc_frames[i]-1,0]
            y_fixed[exc_frames[i]] = body_part_matrix[exc_frames[i]-1,1]
    
    # Return a new bodypart matrix (x,y,conf)
    #return np.transpose(np.asarray((x_fixed, y_fixed, body_part_matrix[:,2])))
    return body_part_matrix

def fix_frames_diff(body_part_matrix,std_threshold):
    ## Get the frames where coordinates have > 0.5*zscore(abs(diff))

    # Get the euclidian distance for each frame
    diff_x = np.diff(body_part_matrix[:,0])
    diff_y = np.diff(body_part_matrix[:,1])
    diff_euc = sts.zscore(np.sqrt(diff_x**2 + diff_y**2))  # Compute the euclidian distance
    mask = diff_euc > 0.5                                  # Get which indices

    exc_frames = np.transpose(np.asarray(np.where(mask == True)))   # Coords excluded (<0.95)
    acc_frames = np.transpose(np.asarray(np.where(mask == False)))  # Coords accepted (>0.95)

    first_acc = acc_frames[0]  # First accepted frame
    last_acc = acc_frames[-1]  # Last accepted frame

    x_fixed = body_part_matrix[:,0] # X coords
    y_fixed = body_part_matrix[:,1] # Y coords

    # Loop for each excluded frame
    for i in range(len(exc_frames)):
        if exc_frames[i] < first_acc: # Frames excluded before the first one accepted
            x_fixed[exc_frames[i]] = body_part_matrix[first_acc,0]
            y_fixed[exc_frames[i]] = body_part_matrix[first_acc,1]
        elif exc_frames[i] > last_acc: # Frames excluded after the last one accepted
            x_fixed[exc_frames[i]] = body_part_matrix[last_acc,0]
            y_fixed[exc_frames[i]] = body_part_matrix[last_acc,1]
        else: # Frames which are inserted between accepted frames
            x_fixed[exc_frames[i]] = body_part_matrix[exc_frames[i]-1,0]
            y_fixed[exc_frames[i]] = body_part_matrix[exc_frames[i]-1,1]
        
    # Return a new bodypart matrix (x,y,conf)
    return np.transpose(np.asarray((x_fixed, y_fixed, body_part_matrix[:,2])))

# def filter_bp_pos_on_maze(bp_pos_on_maze, method_used='complete', win=3, mov_sec=None, fps=30):
#     if method_used == 'complete':
        
#         bp_pos_on_maze_filtered = bp_pos_on_maze.copy()
#         prev_value = bp_pos_on_maze_filtered[0]
#         counter = 1
       
#         for i in range(1, len(bp_pos_on_maze_filtered)):
#             if bp_pos_on_maze_filtered[i] == prev_value:
#                 counter += 1
#             else:
#                 counter = 1
           
#             if counter < win:
#                 bp_pos_on_maze_filtered[i] = prev_value
#             else:
#                 prev_value = bp_pos_on_maze_filtered[i]
       
#         return bp_pos_on_maze_filtered
    
    #     bp_pos_on_maze_filtered = bp_pos_on_maze.copy() # Copy the original array
    #     counter = 1  # Initialize the counter for consecutive occurrences
        
    #     for i in range(1, len(bp_pos_on_maze_filtered)):
    #         if bp_pos_on_maze[i] == bp_pos_on_maze_filtered[i-1]:  # Check if current position is equal to the previous one and add 1 to the counter
    #             counter += 1
    #         else:
    #             counter = 1
            
    #         # If the number of consecutive positions is lesser than the limit window, change the position to a neighboor value
    #         if counter < win:
    #             if i < len(bp_pos_on_maze_filtered) - 1 and bp_pos_on_maze_filtered[i-1] != bp_pos_on_maze_filtered[i+1]:
    #                 bp_pos_on_maze_filtered[i] = bp_pos_on_maze_filtered[i-1]
    #             elif i > 0 and bp_pos_on_maze_filtered[i-1] != bp_pos_on_maze_filtered[i-2]:
    #                 bp_pos_on_maze_filtered[i-1] = bp_pos_on_maze_filtered[i]
    
    # return bp_pos_on_maze_filtered



# Create a moving window to eliminate quick variations of the bp
def filter_bp_pos_on_maze(bp_pos_on_maze, method_used='complete', win=3, mov_sec=None, fps=30):
    # win is the time threshold value between differences to be considered a real diference (in seconds)
    if method_used == 'complete':
        # Create the return value
        bp_pos_on_maze_filtered = bp_pos_on_maze
        # Loop for each possible place in the maze
        for i in np.linspace(12,1,12):
            a = np.transpose(np.array(np.where(bp_pos_on_maze == i)))
            #print(a)
            b = np.transpose(np.array(np.diff(a.T)))
            #print(b)
            mask_logical = np.array((b <= win) & (b > 1))
            mask = np.transpose(np.array(np.where(mask_logical == True))[0,:])
            #mask = np.array(np.where((b <= win) & (b > 1))) # mask to indicate where to change BO TÁ AQUI
            
            for ii in mask:
                idx = ii # Account for diff function
                first = a[idx]+1
                last = a[idx+1]-1
                # # Make sure the last index won't surpass the last value in a
                # if idx+(b[ii]-1) < len(a):
                #     last = np.array(a[idx+(b[ii]-1)])
                # else:
                #     last = np.array(a[-1])
                # if idx+(b[ii]-1) < len(a):
                #     last = np.array(a[idx+(b[ii]-1)])
                # else:
                #     last = np.array(len(a))
              
                #print(idx)
                #print(a[idx])
                #print(a[idx+(b[ii]-1)])
                #bp_pos_on_maze_filtered[a[idx:idx+(b[ii]-1)]] = i  # Insert the correct the position
                bp_pos_on_maze_filtered[np.arange(first,last+1,1)] = i  # Insert the correct the position           
        
    elif method_used == 'movw':
        # If the moving window has been selected as a TIME moving window and not a SAMPLE mov window
        if mov_sec is not None:
            # Calc moving windows based on the moving windows (in seconds)
            mov_win = mov_sec*fps        

        # Call moving average function
        bp_pos_on_maze_filtered = np.round(mov_average(bp_pos_on_maze, mov_win))
    else:
        bp_pos_on_maze_filtered = bp_pos_on_maze
    
    return np.array(bp_pos_on_maze_filtered)


def get_trial_beginning_end_all_bp(body_part_matrix, df, conf_threshold, fps=30, max_duration_trial=3, n_bp=11):
    bps = df.columns.get_level_values(1)    
    
    # Get unique bp values
    indexes = np.unique(bps, return_index=True)[1]  # Get unique
    [bps[index] for index in sorted(indexes)]       # Resort them
    
    # Pre-allocate 
    bpt_cf = np.zeros((len(df),n_bp))
    
    for ii in range(n_bp):  # Loop for each body part
        bpt_cf[:,ii] = df.xs(bps[ii], level='bodyparts', axis=1).to_numpy()[:,2]  # Get confidence interval for each pb

    r_mask, c_mask = np.where(bpt_cf >= conf_threshold)          # Find the frames and bps where confidence interval is above the threshold
    unique, counts = np.unique(r_mask, return_counts=True)    # Get the number of counts
    
    # Define the trial beginning as the first moment when (80% of the animals bp have high confidence interval)
    beg = unique[np.where(counts >= round(n_bp*0,8))[0][0]]
    # Define the trial end as the last moment when (80% of the animals bp have high confidence interval)
    end = unique[np.where(counts >= round(n_bp*0,8))[0][-1]]
    
    # Check if the trial lasted more than the max limit defined by the user
    if (end-beg+1)/fps > max_duration_trial*60:
        end = beg + max_duration_trial*60*fps
   
    
    # Recreate the body_part_matrix with only the frames in between the beginning and end
    body_part_matrix_rec = body_part_matrix[np.arange(beg,end+1,1)]
    #print('n epochs exc: ' + str(beg-1 + np.shape(body_part_matrix)[0]-end+1))
    
    return body_part_matrix_rec, beg, end
    
# Function to extract when the trial begins and when it ends accordingly to the confidence for each frame (The frames with low confidence are ignored: rat inside the white box or escape box)
def get_trial_beginning_end(body_part_matrix, conf_threshold, fps=30, max_duration_trial=3):
    # Find the frames when the conf_threshold is => than the threshold
    mask = np.transpose(np.array(np.where(body_part_matrix[:,2] >= conf_threshold)))
    # First frame
    beg = np.array(mask[0])
    # Last frame
    end = np.array(mask[-1])
    
    # Check if the trial lasted more than the max limit defined by the user
    if (end-beg+1)/fps > max_duration_trial*60:
        end = mask[0] + max_duration_trial*60*fps

    
    # Recreate the body_part_matrix with only the frames in between the beginning and end
    body_part_matrix_rec = body_part_matrix[np.arange(beg,end+1,1)]
    #print('n epochs exc: ' + str(mask[0]-1 + np.shape(body_part_matrix)[0]-mask[-1]+1))
    
    return body_part_matrix_rec

# Function to fix camera shaking during trial recording
def fix_camera_shaking(body_part_matrix,reference_point_matrix, conf_threshold=0.95):
    # Take the last video frame and use the vertices (x,y) references to track any camera movements
    def get_camera_shaking(reference_point_matrix, conf_threshold):
        # Get a confidence mask based on the confidence threshold
        confidence_mask = np.array(np.where(reference_point_matrix[:,2] >= conf_threshold))[0,:]
        selected_frame = confidence_mask[-1]    # Select the last frame above the confidence threshold
        
        reference_point_x = reference_point_matrix[selected_frame,0]    # X value
        reference_point_y = reference_point_matrix[selected_frame,1]    # Y value
        
        x_diff = reference_point_matrix[:,0] - reference_point_x
        y_diff = reference_point_matrix[:,1] - reference_point_y
        
        return reference_point_x, reference_point_y
    
    # Get camera shaking differences on x and y axis
    x_diff, y_diff = get_camera_shaking(reference_point_matrix, conf_threshold)
    
    new_x = body_part_matrix[:,0] - x_diff
    new_y = body_part_matrix[:,1] - y_diff
    new_body_part_matrix = np.vstack((new_x, new_y, body_part_matrix[:,2])).T   # Final fixed body_part_matrix
    
    return new_body_part_matrix


# Fix camera shaking for all the points (nose, head, etc)
def df_fix_camera_shaking(df,bp_reference_str='v_1'):
    
    model = df.columns.get_level_values(0)  # First idx (model)
    bps = df.columns.get_level_values(1)    # Second idx (body part) 
    
    # Get unique bp values
    indexes = np.unique(bps, return_index=True)[1]  # Get unique
    bps = bps[np.arange(0, len(bps), 3)]            
    
    # Get the reference matrix
    reference_point_matrix = df.xs(bp_reference_str, level='bodyparts', axis=1).to_numpy()
    
    # Loop for each body part
    for ii in range(len(bps)):  
        body_part_matrix = df.xs(bps[ii], level='bodyparts', axis=1).to_numpy()  # Get confidence interval for each pb
        # Fixed body part matrix
        new_body_part_matrix = fix_camera_shaking(body_part_matrix,reference_point_matrix, conf_threshold=0.95)
        # Recreate the df with fixed body part matrix
        df[model[0]][bps[ii]] = new_body_part_matrix
    
    # Return fixed df
    return df


def exclude_f_past_duration(df, trial_duration=5, fps=30):
    # Calc the max duration
    max_duration = trial_duration * 60 * round(fps)
    # Get the data frame length
    df_length = df.shape[0]
    
    # Check if the real trial duration is higher than the expected trial duration
    if df_length > max_duration:
        # Exclude the elements that surpass the trial duration
        corrected_df = df.drop(np.arange(max_duration,df_length))
    
    return corrected_df