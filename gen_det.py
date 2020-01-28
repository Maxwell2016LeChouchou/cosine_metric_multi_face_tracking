import os 
import cv2
import numpy as np
import csv

import tensorflow as tf 

import detector
from matplotlib import pyplot as plt 

from PIL import Image




def generate_detections(mot_dir):
    det = detector.face_detection()   

    det_txt = []

    for sequence in os.listdir(mot_dir):
        sequence_dir = os.path.join(mot_dir,sequence)
        img_dir = os.path.join(sequence_dir,"img1")

        det_txt = []
        for image_file in sorted(os.listdir(img_dir)):
            image_name, ext = os.path.splitext(image_file)
            print(image_name)
            images = os.path.join(img_dir,image_file)
        
            img_full = Image.open(images)
            image = det.load_image_into_numpy_array(img_full)
        
            det_bbox = det.get_localization(image, visual=False) # det_bbox = [top, left, width, height]
            #print(det_bbox[1])
            face_id = -1
            confidence = 1 

            det_x = -1
            det_y = -1
            det_z = -1

            for i in range(len(det_bbox)):
                det_txt.append([image_name, face_id, det_bbox[i][0], det_bbox[i][1], det_bbox[i][2], det_bbox[i][3], confidence, det_x, det_y, det_z])

        a = np.array(det_txt) 
        detection_file = os.path.join(sequence_dir,"det/det_0.3.txt")       
        np.savetxt(detection_file, a, fmt="%s,%s,%s,%s,%s,%s,%s,%s,%s,%s")
        
        
    


if __name__ == '__main__':

    frame_dir = '/home/max/Downloads/music_video/GT/test/'   
    generate_detections(frame_dir)
