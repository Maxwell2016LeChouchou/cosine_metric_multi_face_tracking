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

from app_util import preprocessing
from app_util import visualization
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker

import detector as det



def create_detections():
    face boxes = []
    os.chdir(cwd)

    detect_model_name = '/home/max/Desktop/files/ckpt_data_ssd_inception_v2_coco'
    PATH_TO_CKPT = detect_model_name + '/frozen_inference_graph.pb'
    detection_graph = tf.Graph()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
        sess = tf.Session(graph = detection_graph, config=config)
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        scores = detection_graph.get_tensor_by_name('detection_scores:0')
        classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detection:0')
        
        image_expanded = np.expand_dims(image, axis=0)



def run(detection, output_file, min_confidence, nms_max_overlap, min_detection_height, max_cosine_distance, nn_budget, display):

    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)
    results=[]


    def frame_callback(vis, frame_idx):
        detections = det.detection()  # get the detection results from detection pb files
        
        tracker.predict()
        tracker.update(detections)

        #Here we are supposed to have display, but I ingore it since we are doing offline tracking

        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue 
            bbox = track.to_tlwd()
            results.append([bbox[0], bbox[1], bbox[2], bbox[3])

    # Visualize the bbox on faces
    if display:
        images = helpers.draw_box_label(detections, results)


    # Store results
        
    f=open(outfile, 'w')
    for bbox_info in results:
        print('%.2f,%.2f,%.2f,%.2f' %(bbox_info[0], bbox_info[1], bbox_info[2], bbox_info[3]))

        
        

if __name__ == "__main__":
    det = detector.face_detection()

    if debug:
        path_to_test_image_dir = '/home/max/Downloads/MTCNN/multi_face_detection_et_tracking/2/'
