import os 
import numpy as np 
import xml.etree.ElementTree as ET

tree = ET.parse('/home/maxwell/Downloads/GT/tara_gt_test.xml')
root = tree.getroot()

output_dir='/home/maxwell/Downloads/GT/tara_gt.txt'
array_dic = []
for frame in root.iter('Frame'):
    dic = frame.attrib
    
    frame = dic['frame_no']
    id_num = 5
    x = dic['x']
    y = dic['y']
    width = dic['width']
    height = dic['height']
    observation = dic['observation']
    cls_no = 1
    ratio = 1
    array_dic.append(np.array([frame, id_num, x, y, width, height, observation, cls_no, ratio]))
a = np.array(array_dic)
np.savetxt(output_dir,a,fmt="%s,%s,%s,%s,%s,%s,%s,%s,%s")
print(output_dir)





    
    # for x, y in enumerate(frame.attrib):
    #     print(x)
