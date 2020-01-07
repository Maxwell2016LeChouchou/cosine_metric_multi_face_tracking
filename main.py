from __future__ import division, print_function, absolute_import

import argparse
import os

import cv2
import numpy as np

import tensorflow as tf 
from PIL import Image
import sys
from matplotlib import pyplot as plt 
import time 
from glob import glob 

from application_util import preprocessing
from application_util import visualization
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker

from moviepy.editor import VideoFileClip

import detector

from application_util import visual_helpers


# def create_detections():

#     det = detector.face_detection()
#     detection_boxs = det.get_localization(img)
#     #for row, detections in enumerate(detection_boxs):
        
    
#     return detection_boxs


#def run(img, output_file, min_confidence, 
    #min_detection_height, max_cosine_distance, nn_budget, display):

def run(img):
    output_file='/home/max/Desktop/yt_test_data/output_file/track_results.txt'

    #min_confidence = 0.3
    #min_detection_height = 0 
    max_cosine_distance = 0.2
    nn_budget = None
    display = True

    metric = nn_matching.NearestNeighborDistanceMetric(
        "cosine", max_cosine_distance, nn_budget)  # Here they set lamda as zero
    tracker = Tracker(metric)
    results=[]

    det = detector.face_detection()
    detections = det.get_localization(img)

    tracker.predict()
    tracker.update(detections)

    for track in tracker.tracks:
        if not track.is_confirmed() or track.time_since_update > 1:
            continue
        bbox = track.to_tlbr()
        results.append([bbox[0], bbox[1], bbox[2], bbox[3]])
        
        if display:
            img = visual_helpers.draw_box_label(img,bbox)
        else:
            print("No display")
    


    # Store results
    f=open(output_file, 'w')
    for bbox_info in results:
        print('%.2f,%.2f,%.2f,%.2f' %(bbox_info[0], bbox_info[1], bbox_info[2], bbox_info[3]),file=f)

    return img


# def run(img, output_file, min_confidence, 
#     min_detection_height, max_cosine_distance, 
#     nn_budget, display):

#     metric = nn_matching.NearestNeighborDistanceMetric(
#         "cosine", max_cosine_distance, nn_budget)  # Here they set lamda as zero
#     tracker = Tracker(metric)
#     results=[]


#     def frame_callback(vis, frame_idx):

#         # Obtain the detection results
#         det = detector.face_detection()
#         detections = det.get_localization(img)

#         """
#         Here original has the NMS but we dont need this
#         """
        
#         # Update tracker
#         tracker.predict()
#         tracker.update(detections)

#         # Update visualization
#         if display:
#             image = cv2.imread(img) 
#             vis.set_image(image.copy())
#             vis.draw_detections(detections)
#             vis.draw_trackers(tracker.tracks)
            

#         # Store results
#         for track in tracker.tracks:
#             if not track.is_confirmed() or track.time_since_update > 1:
#                 continue 
#             bbox = track.to_tlwh()
#             results.append([bbox[0], bbox[1], bbox[2], bbox[3]])

#     # Run tracker
#     if display:
#         visualizer = visualization.Visualization(img, update_ms=5)
#     else:
#         print("No visualization")
#     visualizer.run(frame_callback)


#     # Store results
#     f=open(output_file, 'w')
#     for bbox_info in results:
#         print('%.2f,%.2f,%.2f,%.2f' %(bbox_info[0], bbox_info[1], bbox_info[2], bbox_info[3]),file=f)




def bool_string(input_string):
    if input_string not in {"True", "False"}:
        raise ValueError("Please Enter a valid True/False choice")
    else: 
        return (input_string == "True")  


if __name__ == "__main__":


    output_video = '/home/max/Desktop/yt_test_data/output_file/test_1.mp4'

    video_file = "/home/max/Downloads/MTCNN/multi_face_detection_et_tracking/maxwell_lingfeng.mp4"
    clip1 = VideoFileClip(video_file)
    clip = clip1.fl_image(run)
  
    clip.write_videofile(output_video, audio=False)


    #clip = cv2.VideoCapture(video_file)
    
    #rval, frame=clip.read()

    #run(frame, output_file, min_confidence, 
        #min_detection_height, max_cosine_distance, 
        #nn_budget, display)
        