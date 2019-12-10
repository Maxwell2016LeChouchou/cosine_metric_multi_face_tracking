from __future__ import division, print_function, absolute_import

import argparse
import os

import cv2
import numpy as np 

from app_util import preprocessing
from app_util import visualization
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker

import detector as det


def run(detection, output_file, min_confidence, nms_max_overlap, min_detection_height, max_cosine_distance, nn_budget, display):

    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)
    results=[]


    def fram_callback(vis, frame_idx):
        detections = det.detection()  # get the detection results from detection pb files
        





if __name__ == "__main__":
    det = detector.face_detection()

    if debug:
        path_to_test_image_dir = '/home/max/Downloads/MTCNN/multi_face_detection_et_tracking/2/'
