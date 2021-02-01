# import libraries
import numpy as np
import glob
import datetime
import cv2
import subprocess
import os

def get_file_names_from_dir(dir_path, ext_str):
    return glob.glob(dir_path + ext_str)

def make_directory(input_directory):    
    if not os.path.exists(input_directory):
        os.makedirs(input_directory)

# set mini segment length
interval = 30

# set input & output directory
# note: need to modify directory for prod
base_path = '/Users/carechen/poc2/'
video_path = base_path + 'classifier_main_content/'
clip_path = base_path + 'classifier_clip/'
output = clip_path + 'mini_clips_' + str(interval) + 's.txt'

# get files
files = get_file_names_from_dir(video_path, '*.mp4')  
all_clips = []
for f in files:
    all_clips.append(f)
all_clips.sort()
print(len(all_clips),all_clips)

# segment main content into mini segments
result = []
st_sec = 600 # set start skip seconds to avoid open credit and intro/recap
ed_skip_ratio = 0.2 # set end skip ratio to avoid spoiler and end credits

for f in files:
    # get video metadata and find number segments
    cap = cv2.VideoCapture(f)
    fps = cap.get(cv2.CAP_PROP_FPS)      
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = frame_count / fps 
    ed_sec = video_duration * ed_skip_ratio # set end skip seconds based on video duration and end skip ratio
    process_duration = video_duration - st_sec - ed_sec
    seg_nbr = int(process_duration/interval) + 1
    print('fps, frame count = ', fps, frame_count) 
    print('video_duration, start seconds to skip, end seconds to skip = ', video_duration, st_sec, ed_sec)
    print('process_duration, interval, seg_nbr = ', process_duration, interval, seg_nbr)
    
    # find timestamps associated with mini clip
    timestamps = []
    for i in range(seg_nbr):
        str_ts0 = str(datetime.timedelta(seconds = (st_sec + i*interval)))
        # check whether the format is '%H:%M:%S' or '%H:%M:%S.%f', and append segment timestamps
        if(len(str_ts0.split('.')) == 2):
            str_ts = str_ts0interval
        else:
            str_ts = str_ts0 + '.000000'
        timestamps.append(str_ts)
    # add last segment timestamps
    end_ts0 = str(datetime.timedelta(seconds = (st_sec + process_duration)))
    if(len(end_ts0.split('.')) == 2):
        end_ts = end_ts0
    else:
        end_ts = end_ts0 + '.000000'
    timestamps.append(end_ts0)
    print('mini clip timestamps = ', timestamps)
    
    # generate mini-segment clips
    make_directory(clip_path)
    
    for j in range(seg_nbr): # get all selected mini clips
        lf, rt = timestamps[j], timestamps[j+1]
        #print('lf = ', lf)
        #print('rt = ', rt)
        start_time = lf
        delta = str(datetime.datetime.strptime(rt, "%H:%M:%S.%f") - datetime.datetime.strptime(lf, "%H:%M:%S.%f"))
        in_file = f
        cur_name = f[45:-4] # change index based on clip directory
        out_file = clip_path + cur_name + '_' + 'clip_' + str(interval) + 'sec_' + str(j).zfill(6) + '.mp4'
        print(start_time,delta,out_file)
        result.append(out_file)
        subprocess.call([ 'ffmpeg', '-loglevel', 'panic', '-ss', start_time,  '-t', delta, '-i', in_file,  '-c', 'copy' , '-y', out_file])            
        #cmd = 'ffmpeg -i ' + in_file + ' -ss ' + start_time + ' -strict -2 -t ' + delta + ' -b:v 1M ' + out_file
        #os.system(cmd)

# write out_file to txt for class prediction
txt = open(output, 'w')
for row in result:
    txt.write(str(row)+'\n')
txt.close()


