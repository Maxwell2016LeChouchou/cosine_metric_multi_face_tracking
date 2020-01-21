import cv2
vidcap = cv2.VideoCapture('/home/maxwell/Downloads/Darling.mp4')
success,image = vidcap.read()
count = 1
while success:
  cv2.imwrite("/home/maxwell/Downloads/output/Darling/%d.jpg" % count, image)     # save frame as JPEG file      
  success,image = vidcap.read()
  #print('Read a new frame: ', success)
  count += 1