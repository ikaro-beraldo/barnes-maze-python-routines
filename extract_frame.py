import cv2
import ffmpeg

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

def get_video_fps(video_filename):
    # Get the video filename based on the H5 DLC output name
    check_vid_filename = video_filename[0:video_filename.index('DLC')] + '.mp4'
    
    # Use ffmpeg to probe the video file
    try:
        probe = ffmpeg.probe(check_vid_filename)
    except ffmpeg.Error as e:
        print(e.stderr)

    # Create a dict to organize the video info (everything is str so, it has to be extract as int)
    video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
    fps1,fps2 = video_info['r_frame_rate'].split('/')
    # Get FPS value
    fps = int(fps1)/int(fps2)
    
    return fps