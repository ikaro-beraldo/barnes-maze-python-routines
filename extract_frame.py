import cv2

def extract_frame_f_video(video_filename, video_frame=None, fps=30):
    vidcap = cv2.VideoCapture(video_filename)  # Video capture
    ret = False
    
    # GET ONLY THE FIRST FRAME CASE THE VIDEO FRAME HAS NOT BEEN SELECTED
    if video_frame is None:
        while ret==False:
            # read the first video frame
            ret,frame = vidcap.read()
        
    # IF A SPECIFIC VIDEO FRAME HAS BEEN GIVEN 
    else:
        while ret==False:
            time_msec = (video_frame/fps)*1000     # Multiple by 1000 to get it in milliseconds
            vidcap.set(cv2.CAP_PROP_POS_MSEC,time_msec)      # just cue to 20 sec. position
            ret,frame = vidcap.read()
        
    # Return the frame as image
    return cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)