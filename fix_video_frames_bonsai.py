# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 15:25:48 2023

@author: ikaro
"""

# Fix the frame rate from videos recorded on bonsai

import cv2


cap = cv2.VideoCapture("C:\\Users\\ikaro\\Downloads\\Imp15_Aquisição.avi")


## Get video metadata
video_fps = cap.get(cv2.CAP_PROP_FPS),
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)

## Get video new info
n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
n_frames = vidObj.NumFrames;    % number of frames
frame_rate = vidObj.FrameRate;  % frame rate
expected_frame_rate = 50;
expected_n_frames = round((expected_frame_rate/frame_rate)*n_frames);

% New frames vector
new_frames = round(linspace(1,n_frames,expected_n_frames));


## we are using x264 codec for mp4
fourcc = cv2.VideoWriter_fourcc(*'X264')
writer = cv2.VideoWriter('OUTPUT_PATH.mp4', apiPreference=0, fourcc=fourcc,
                     fps=video_fps[0], frameSize=(int(width), int(height)))

while True:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number-1)
    ret, frame = cap.read()
    if not ret: break # break if cannot receive frame
    # convert to grayscale
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    writer.write(frame) # write frame
    
    if cv2.waitKey(1) & 0xFF == ord('q'): # on press of q break
        break
        
# release and destroy windows
writer.release()
cap.release()
cv2.destroyAllWindows()