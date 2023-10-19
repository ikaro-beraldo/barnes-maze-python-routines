# Set of functions to plot stuff

from matplotlib import pyplot as plt
from data_handling import *
from extract_frame import *
from color_functions import *
import matplotlib.pyplot as plt
import imageio
from scipy.ndimage.filters import gaussian_filter
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from matplotlib import patches
import numpy as np

# Recreate the Barnes Maze in a previous defined axes
def maze_recreation_plot(axes, body_part_matrix, centroid_coords, position_each_hole, maze_info_pixel, plot_frame=False, title='Nose', show=True, invert=True):
    
    # If the user has selected to plot a video frame
    if plot_frame is True:
        # Get the video file regarding the coordinates file IN THE FUTURE: READ VIDEO AUTOMATICALLY 
        dh = Data_handling
        filename = dh.select_file()
        # Get a video frame
        frame = extract_frame_f_video(filename)
        # Plot the frame
        axes.imshow(frame)
    else: # If there is not a video frame
        # Redefine the plot limits
        axes.set_xlim((centroid_coords[0] - maze_info_pixel["maze_radius_pixels"] - 100, centroid_coords[0] + maze_info_pixel["maze_radius_pixels"] + 100))
        axes.set_ylim((centroid_coords[1] - maze_info_pixel["maze_radius_pixels"] - 100, centroid_coords[1] + maze_info_pixel["maze_radius_pixels"] + 100))
        
        if invert is True:
            # Invert the yaxis
            axes.invert_yaxis()
    
    # Plot the Maze circunference
    maze_circle = plt.Circle((centroid_coords[0],centroid_coords[1]), radius=maze_info_pixel["maze_radius_pixels"], fill=True, facecolor='black', alpha=0.1)
    axes.set_aspect(1) 
    axes.add_artist(maze_circle) 
    
    # Plot the hole positions
    for i in range(12):
        circle = plt.Circle((position_each_hole[i,0], position_each_hole[i,1]), maze_info_pixel["hole_radius_pixels"], color='red', alpha=0.2)
        axes.add_patch(circle)
    
    #axes.plot(position_each_hole[:,0], position_each_hole[:,1], '.r', markersize=maze_info_pixel["hole_radius_pixels"], alpha=0.2)
    
    # Plot the centroid 
    axes.plot(centroid_coords[0], centroid_coords[1], 'og')
    
    ## Y_tick and X_tick behavior
    locs = np.asarray(plt.xticks()[0])
    
    ###################################### PERFUMARIAS ###########################################
    ## Remove the plot ticks
    axes.set_xticks([])
    axes.set_yticks([])
    # Create a scale line aproximately 25 cm
    # line y
    x1 = (centroid_coords[0] + maze_info_pixel["maze_radius_pixels"]) - (25* maze_info_pixel["pixelcm_ratio"])
    x2 = centroid_coords[0] + maze_info_pixel["maze_radius_pixels"]
    y = centroid_coords[1] + maze_info_pixel["maze_radius_pixels"]
    y = np.array([y, y])
    axes.plot(np.array([x1, x2]),y, linewidth=2.5, color='black')
    axes.text(x1+25, y[0]-10, "25 cm")
    axes.set_title(title, fontdict=None, loc='center')
    ###############################################################################################
    
    # Important case any other data is going to be ploted on the maze
    if show is True:
        plt.show()

def plot_trajectory(axes, body_part_matrix, fps=30):
    # Trajectory itself
    c = colorline(body_part_matrix[:,0], body_part_matrix[:,1], cmap="cool", ax=axes)
    cbar = plt.colorbar(c,ax=axes, ticks=[0, 0.5, 1])
    # Get the max, half and min values of elapsed time
    time_ticks = [str(0), str(round(np.shape(body_part_matrix)[0]/2/fps,2)), str(round(np.shape(body_part_matrix)[0]/fps,2))]
    cbar.ax.set_yticklabels(time_ticks)
    cbar.ax.set_ylabel('Time elapsed')    
        

def plot_animal_bp_position(axes, bp_pos_on_maze, trial_data=None, fps=30):
    
    colorline(np.arange(len(bp_pos_on_maze))/fps, bp_pos_on_maze, cmap="cool", linewidth=1.5, ax=axes)
    axes.set_xlim((0,len(bp_pos_on_maze)/fps))
    axes.set_ylim((-2,13))
    axes.set_yticks(np.arange(-1,13,1), ['Out','On','#1','#2','#3','#4','#5','#6','#7','#8','#9','#10','#11','#12'])
    axes.set_xlabel("Seconds")
    axes.set_ylabel("BP position")
    axes.set_title("Body part position on maze")
    #plt.show()
    
    if not(trial_data is None):
        # Plot a text with the number of errors, and strategy
        axes.text(len(bp_pos_on_maze)/fps/50, 12, '1º errors: '+str(trial_data['P_error'][0]))
        axes.text(len(bp_pos_on_maze)/fps/50, 11, '2º errors: '+str(trial_data['S_error'][0]))
        axes.text(len(bp_pos_on_maze)/fps/50, 10, 'Strategy: '+trial_data['Strategy'][0])
        
def plot_object_exploration(axes, obj_exploration, trial_data=None, fps=30):
    
    colorline(np.arange(len(obj_exploration))/fps, obj_exploration, cmap="cool", linewidth=1.5, ax=axes)
    axes.set_xlim((0,len(obj_exploration)/fps))
    axes.set_ylim((-0.5,3.5))
    axes.set_yticks(np.arange(0,3,1), ['On','Obj #1','Obj #2'])
    axes.set_xlabel("Seconds")
    axes.set_ylabel("BP position")
    axes.set_title("Body part position on maze")
    #plt.show()
    
    if not(trial_data is None):
        # Plot a text with the number of errors, and strategy
        axes.text(len(obj_exploration)/fps/50, 3.25, 'Obj 1: '+str(trial_data['time_obj_1'][0])+' Ratio: '+str(trial_data['ratio_1_total'][0]))
        axes.text(len(obj_exploration)/fps/50, 3, 'Obj 2: '+str(trial_data['time_obj_2'][0])+' Ratio: '+str(trial_data['ratio_2_total'][0]))
        axes.text(len(obj_exploration)/fps/50, 2.75, 'Total: '+str(trial_data['time_obj_1'][0]+trial_data['time_obj_2'][0]))
        
def plot_instant_speed_maze(axes, body_part_matrix, inst_speed_entire, centroid_coords, position_each_hole, maze_info_pixel):
    
    # Plot the maze recreation
    maze_recreation_plot(axes, body_part_matrix, centroid_coords, position_each_hole, maze_info_pixel, plot_frame=False, title='Body Centre', show=False)
    # Show plots on the selected axes    
    c = colorline(body_part_matrix[:,0], body_part_matrix[:,1], z=inst_speed_entire/inst_speed_entire.max(), cmap="magma", linewidth=1.5, ax=axes)
    cbar = plt.colorbar(c,ax=axes, ticks=[0, 0.5, 1])
    # Get the max, half and min speeds
    speed_ticks = [str(np.floor(np.min(inst_speed_entire))), str(np.floor(np.max(inst_speed_entire))*0.5), str(np.floor(np.max(inst_speed_entire)))]
    cbar.ax.set_yticklabels(speed_ticks)
    cbar.ax.set_ylabel('cm/s')
    
def plot_instant_speed_maze_OLR(axes, body_part_matrix, inst_speed_entire, centroid_coords, position_each_vertex, maze_info_pixel):
    
    # Plot the maze recreation
    maze_recreation_plot_OLR(axes, centroid_coords, position_each_vertex, maze_info_pixel, plot_frame=False, title='Body Centre', show=False, invert=False)
    # Show plots on the selected axes    
    c = colorline(body_part_matrix[:,0], body_part_matrix[:,1], z=inst_speed_entire/inst_speed_entire.max(), cmap="magma", linewidth=1.5, ax=axes)
    cbar = plt.colorbar(c,ax=axes, ticks=[0, 0.5, 1])
    # Get the max, half and min speeds
    speed_ticks = [str(np.floor(np.min(inst_speed_entire))), str(np.floor(np.max(inst_speed_entire))*0.5), str(np.floor(np.max(inst_speed_entire)))]
    cbar.ax.set_yticklabels(speed_ticks)
    cbar.ax.set_ylabel('cm/s')

    
def plot_instant_speed(axes, inst_speed_entire, body_part_matrix, trial_data=None, fps=30, time_window=10):
    
    x = np.linspace(0,len(inst_speed_entire)/fps,len(inst_speed_entire))    # Produce x axis
    c = colorline(x, inst_speed_entire, z=inst_speed_entire/inst_speed_entire.max(), cmap="magma", linewidth=1.5, ax=axes)
    axes.set_xlim((0,len(inst_speed_entire)/fps))
    axes.set_ylim((0,100))
    #axes.set_yticks(np.arange(-1,13,1), ['Out','On','#1','#2','#3','#4','#5','#6','#7','#8','#9','#10','#11','#12'])
    axes.set_xlabel("Seconds")
    axes.set_ylabel("cm/s")
    bin_length = time_window/fps
    axes.set_title("Instant speed - "+str(round(bin_length,2)) + " sec bin")
    
    if not(trial_data is None):
        # Plot a text with the average speed
        axes.text(len(inst_speed_entire)/fps/50, 90, 'Average Speed: '+str(round(trial_data['Av_speed'][0],2))+'cm/s')
        # Plot the average speed as an black line
        axes.axhline(y = trial_data['Av_speed'][0], xmin = 0, xmax = len(inst_speed_entire)/fps, color = 'k')
     

# Plot the barnes maze and animal position throughout trial as pdf
def plot_animal_route_as_pdf(filename, body_part_matrix, centroid_coords, position_each_hole, maze_info_pixel, plot_frame=False, title='Nose', show=False):
    figure, axes = plt.subplots()
    maze_recreation_plot(axes, body_part_matrix, centroid_coords, position_each_hole, maze_info_pixel, plot_frame=False, title='Nose', show=False)
    # Show plots on the selected axes    
    colorline(body_part_matrix[:,0], body_part_matrix[:,1], cmap="cool", linewidth=2)

    plt.savefig(filename, format="pdf", bbox_inches="tight", dpi=300) 
    plt.show()        
        

# Create the heat map
def heatmap(x, y, s, plt_range, bins=200):
    heatmap, xedges, yedges = np.histogram2d(x, y, bins=bins, range=plt_range)
    heatmap = gaussian_filter(heatmap, sigma=s)

    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    return heatmap.T, extent

# Function to call the heatmap and plot it
def plot_heatmap(body_part_matrix, centroid_coords, position_each_hole, maze_info_pixel, ax=None):
    
    plot_range = [[centroid_coords[0] - maze_info_pixel["maze_radius_pixels"] - 100, centroid_coords[0] + maze_info_pixel["maze_radius_pixels"] + 100], [centroid_coords[1] - maze_info_pixel["maze_radius_pixels"] - 100, centroid_coords[1] + maze_info_pixel["maze_radius_pixels"] + 100]]
    
    # Get the heat map info
    hmap, extent = heatmap(body_part_matrix[:,0], body_part_matrix[:,1], 5, plt_range=plot_range)
    # Get maze info
    alphas = np.clip(Normalize(0, hmap.max(), clip=True)(hmap)*1.5, 0.0, 1.)
    colors = Normalize(0, hmap.max(), clip=True)(hmap)
    colors = cm.magma(colors)
    colors[..., -1] = alphas
    
    if ax is None:
        fig, ax = plt.subplots()
        
    # Plot Maze
    maze_recreation_plot(ax, body_part_matrix, centroid_coords, position_each_hole, maze_info_pixel, plot_frame=False, title='Body Centre', show=False, invert=False)
    ax.invert_yaxis()
    
    #ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])
    #ax.imshow(bg)
    h = ax.imshow(np.flip(colors, axis=0), extent=extent, cmap='magma', alpha=1)
    cbar = plt.colorbar(h,ax=ax, ticks=[0, 0.5, 1])
    # Get the max, average and min time spent based on hmap values
    max_value = hmap.max()
    time_spent_ticks = [str(0), str(round(max_value*0.5,2)), str(round(max_value,2))]
    cbar.ax.set_yticklabels(time_spent_ticks)
    cbar.ax.set_ylabel('seconds')
    #plt.scatter(body_part_matrix[:,0], body_part_matrix[:,1])
    
# Function to call the heatmap and plot it
def plot_heatmap_OLR(body_part_matrix, centroid_coords, position_each_vertex, maze_info_pixel, ax=None):
    
    plot_range = [[position_each_vertex[0,0] - 100, position_each_vertex[1,0] + 100], [position_each_vertex[0,1] - 100, position_each_vertex[3,1] + 100]]
    
    # Get the heat map info
    hmap, extent = heatmap(body_part_matrix[:,0], body_part_matrix[:,1], 5, plt_range=plot_range)
    # Get maze info
    alphas = np.clip(Normalize(0, hmap.max(), clip=True)(hmap)*1.5, 0.0, 1.)
    colors = Normalize(0, hmap.max(), clip=True)(hmap)
    colors = cm.magma(colors)
    colors[..., -1] = alphas
    
    if ax is None:
        fig, ax = plt.subplots()
        
    # Plot Maze
    maze_recreation_plot_OLR(ax, centroid_coords, position_each_vertex, maze_info_pixel, plot_frame=False, title='Body Centre', show=False, invert=False)
    ax.invert_yaxis()
    
    #ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])
    #ax.imshow(bg)
    h = ax.imshow(np.flip(colors, axis=0), extent=extent, cmap='magma', alpha=1)
    cbar = plt.colorbar(h,ax=ax, ticks=[0, 0.5, 1])
    # Get the max, average and min time spent based on hmap values
    max_value = hmap.max()
    time_spent_ticks = [str(0), str(round(max_value*0.5,2)), str(round(max_value,2))]
    cbar.ax.set_yticklabels(time_spent_ticks)
    cbar.ax.set_ylabel('seconds')
    #plt.scatter(body_part_matrix[:,0], body_part_matrix[:,1])
    
def plot_quadrants(quadrant_dict, fps=30, ax=None):
    
    for ii in range(len(quadrant_dict)):
        # Check if the body part is positioned on the specific
        triangle = quadrant_dict['quadrant'+str(ii+1)]    
        
        # Plot the lines separating the quadrants 
        ax.plot(np.array([triangle[0,0], triangle[2,0]]), np.array([triangle[0,1], triangle[2,1]]), color='black', alpha=0.2)
    
    # Invert the yaxis
    #plt.gca().invert_yaxis()
    #plt.show()
    
def plot_quadrants_OLR(quadrant_dict, fps=30, ax=None):
    
    for ii in range(len(quadrant_dict)):
        # Check if the body part is positioned on the specific
        rectangle = quadrant_dict['quadrant'+str(ii+1)]    
        
        # Plot the lines separating the quadrants 
        ax.plot(np.array([rectangle[0,0], rectangle[1,0]]), np.array([rectangle[0,1], rectangle[1,1]]), color='black', alpha=0.2)
    
    # Invert the yaxis
    #plt.gca().invert_yaxis()
    #plt.show()
    
def plot_animal_quadrant_position(axes, bp_pos_on_quadrant, fps=30):
    
    colorline(np.arange(len(bp_pos_on_quadrant))/fps, bp_pos_on_quadrant, cmap="magma", linewidth=1.5, ax=axes)
    axes.set_xlim((0,len(bp_pos_on_quadrant)/fps))
    axes.set_ylim((0,5))
    axes.set_yticks(np.arange(1,5,1), ['#1','#2','#3','#4'])
    axes.set_xlabel("Seconds")
    axes.set_ylabel("Quadrant")
    axes.set_title("Body part position by quadrant")
    

# Create a plot (2,3) with the main informations regarding the trial
def big_plot_OLR(body_part_matrix_nose,body_part_matrix_body_centre, obj_exploration, centroid_coords, position_each_vertex, maze_info_pixel, inst_speed_entire, quadrant_dict, bp_pos_on_quadrant, save_filename, trial_data=None, trial_name=None, show=True, fps=30):
    figure, axes = plt.subplots(3,2, figsize=(12.8, 9.6))  # Define the 6 subplots
       
    # 1st PLOT --> the maze and the body part trajectory
    # Recreate the maze
    maze_recreation_plot_OLR(axes[0,0], centroid_coords, position_each_vertex, maze_info_pixel, plot_frame=False, title='Nose', show=False, invert=False)
    # BP trajectory
    plot_trajectory(axes[0,0], body_part_matrix_nose, fps=fps)
    axes[0,0].invert_yaxis()
    
    # 2nd PLOT --> the body part position considering holes and regions
    plot_object_exploration(axes[0,1], obj_exploration, trial_data=trial_data, fps=fps)
    
    # 3rd PLOT --> plot the animal instant speed on maze
    plot_instant_speed_maze_OLR(axes[1,0], body_part_matrix_body_centre, inst_speed_entire, centroid_coords, position_each_vertex, maze_info_pixel)
    axes[1,0].invert_yaxis()

    # 4th PLOT --> plot the animal instant speed on time
    plot_instant_speed(axes[1,1], inst_speed_entire, body_part_matrix_body_centre, trial_data=trial_data, fps=fps)
    
    # 5th PLOT --> plot the animal spent on a specif part of the maze as a heat map
    plot_heatmap_OLR(body_part_matrix_body_centre, centroid_coords, position_each_vertex, maze_info_pixel, ax=axes[2,0])
    plot_quadrants_OLR(quadrant_dict, ax=axes[2,0], fps=fps)
    
    # 6th PLOT --> plot the animal position considering each one of the 4 quadrants
    plot_animal_quadrant_position(axes[2,1], bp_pos_on_quadrant, fps=fps)
    
    # Insert the trial name as the figure main title
    if not(trial_name is None):
        figure.suptitle(trial_name)

    # Make sure every subplot is tightly adjusted
    plt.tight_layout()
    # Save figure as pdf
    plt.savefig(save_filename+'.pdf', format="pdf", bbox_inches="tight", dpi=300) 
    plt.savefig(save_filename+'.png', format="png", bbox_inches="tight", dpi=300) 
    if show is True:
        plt.show()
    
    return axes

# Create a plot (2,3) with the main informations regarding the trial
def big_plot(body_part_matrix_nose,body_part_matrix_body_centre, bp_pos_on_maze, centroid_coords, position_each_hole, maze_info_pixel, inst_speed_entire, quadrant_dict, bp_pos_on_quadrant, save_filename, trial_data=None, trial_name=None, show=True):
    figure, axes = plt.subplots(3,2, figsize=(12.8, 9.6))  # Define the 6 subplots
       
    # 1st PLOT --> the maze and the body part trajectory
    # Recreate the maze
    maze_recreation_plot(axes[0,0], body_part_matrix_nose, centroid_coords, position_each_hole, maze_info_pixel, plot_frame=False, title='Nose', show=False, invert=True)
    # BP trajectory
    plot_trajectory(axes[0,0], body_part_matrix_nose, fps=30)
    
    # 2nd PLOT --> the body part position considering holes and regions
    plot_animal_bp_position(axes[0,1], bp_pos_on_maze, trial_data=trial_data)
    
    # 3rd PLOT --> plot the animal instant speed on maze
    plot_instant_speed_maze(axes[1,0], body_part_matrix_body_centre, inst_speed_entire, centroid_coords, position_each_hole, maze_info_pixel)

    # 4th PLOT --> plot the animal instant speed on time
    plot_instant_speed(axes[1,1], inst_speed_entire, body_part_matrix_body_centre, trial_data=trial_data)
    
    # 5th PLOT --> plot the animal spent on a specif part of the maze as a heat map
    plot_heatmap(body_part_matrix_body_centre, centroid_coords, position_each_hole, maze_info_pixel, ax=axes[2,0])
    plot_quadrants(quadrant_dict, ax=axes[2,0])
    
    # 6th PLOT --> plot the animal position considering each one of the 4 quadrants
    plot_animal_quadrant_position(axes[2,1], bp_pos_on_quadrant, fps=30)
    
    # Insert the trial name as the figure main title
    if not(trial_name is None):
        figure.suptitle(trial_name)

    # Make sure every subplot is tightly adjusted
    plt.tight_layout()
    # Save figure as pdf
    plt.savefig(save_filename+'.pdf', format="pdf", bbox_inches="tight", dpi=300) 
    plt.savefig(save_filename+'.png', format="png", bbox_inches="tight", dpi=300) 
    if show is True:
        plt.show()
    
    return axes
    
    
# Recreate the Barnes Maze in a previous defined axes
def maze_recreation_plot_OLR(axes, centroid_coords, position_each_vertex, maze_info_pixel, plot_frame=True, title='Nose', show=True, invert=True, video_file=None, video_frame=None):
    
    # If the user has selected to plot a video frame
    if plot_frame is True:
        # If the video file has not already been selected
        if video_file is None:
            # Get the video file regarding the coordinates file IN THE FUTURE: READ VIDEO AUTOMATICALLY 
            filename = select_file()
            # Get a video frame
            frame = extract_frame_f_video(filename[0])
        else:
            filename = video_file
            frame = extract_frame_f_video(filename, video_frame=video_frame)
        
        # Plot the frame
        axes.imshow(frame)
    else: # If there is not a video frame
        # Redefine the plot limits
        axes.set_xlim((position_each_vertex[0,0] - 100, position_each_vertex[1,0] + 100))
        axes.set_ylim((position_each_vertex[0,1] - 100, position_each_vertex[3,1] - 100))
        
        if invert is True:
            # Invert the yaxis
            axes.invert_yaxis()
    
    # Plot the Maze rectangles
    # Plot the interior rectangles
    axes.plot(position_each_vertex[[0,1,2,3,0],0],position_each_vertex[[0,1,2,3,0],1], 'go-', linewidth=2)
    axes.plot(position_each_vertex[[4,5,6,7,4],0],position_each_vertex[[4,5,6,7,4],1], 'go-', linewidth=2)
    
    # Plot the objects positions   
    # Obj 1
    x = maze_info_pixel['av_g_1'][0]
    y = maze_info_pixel['av_g_1'][1]
    ellipse = patches.Ellipse((x,y), maze_info_pixel["emp_radius_pixel"][0]*2, maze_info_pixel["emp_radius_pixel"][1]*2, angle=maze_info_pixel['angle_g_1'], color='red', alpha=0.2)
    ellipse_outer = patches.Ellipse((x,y), maze_info_pixel["emp_radius_pixel_outer"][0]*2, maze_info_pixel["emp_radius_pixel_outer"][1]*2, angle=maze_info_pixel['angle_g_1'], color='blue', alpha=0.2)
    axes.add_patch(ellipse)
    axes.add_patch(ellipse_outer)
    
    # Obj 2
    x = maze_info_pixel['av_g_2'][0]
    y = maze_info_pixel['av_g_2'][1]
    ellipse = patches.Ellipse((x,y), maze_info_pixel["emp_radius_pixel"][0]*2, maze_info_pixel["emp_radius_pixel"][1]*2, angle=maze_info_pixel['angle_g_2'], color='red', alpha=0.2)
    ellipse_outer = patches.Ellipse((x,y), maze_info_pixel["emp_radius_pixel_outer"][0]*2, maze_info_pixel["emp_radius_pixel_outer"][1]*2, angle=maze_info_pixel['angle_g_2'], color='blue', alpha=0.2)
    axes.add_patch(ellipse)
    axes.add_patch(ellipse_outer)
    
    #axes.plot(position_each_vertex[:,0], position_each_vertex[:,1], '.r', markersize=maze_info_pixel["vertex_radius_pixels"], alpha=0.2)
    
    # Plot the centroid 
    axes.plot(centroid_coords[0], centroid_coords[1], 'og')
    
    ## Y_tick and X_tick behavior
    locs = np.asarray(plt.xticks()[0])
    
    ###################################### PERFUMARIAS ###########################################
    # Set the x and y limits
    axes.set_xlim((position_each_vertex[0,0] - 100, position_each_vertex[1,0] + 100))
    axes.set_ylim((position_each_vertex[0,1] - 100, position_each_vertex[3,1] + 100))
        
    ## Remove the plot ticks
    axes.set_xticks([])
    axes.set_yticks([])
    # Create a scale line aproximately 25 cm
    # line y
    # x1 = (centroid_coords[0] + maze_info_pixel["maze_radius_pixels"]) - (25* maze_info_pixel["pixelcm_ratio"])
    # x2 = centroid_coords[0] + maze_info_pixel["maze_radius_pixels"]
    # y = centroid_coords[1] + maze_info_pixel["maze_radius_pixels"]
    # y = np.array([y, y])
    # axes.plot(np.array([x1, x2]),y, linewidth=2.5, color='black')
    # axes.text(x1+25, y[0]-10, "25 cm")
    # axes.set_title(title, fontdict=None, loc='center')
    ###############################################################################################
    
    # Important case any other data is going to be ploted on the maze
    if show is True:
        plt.show()
   
