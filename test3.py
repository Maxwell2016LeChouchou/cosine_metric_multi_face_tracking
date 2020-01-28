import cv2
vidcap = cv2.VideoCapture('/home/maxwell/Downloads/Apink.mp4')
success,image = vidcap.read()
count = 1
while success:
  count_str = str(count)
  while len(count_str) < 4:
    count_str = "0" + count_str
  cv2.imwrite("/home/maxwell/Downloads/music_video/apink/Apink/img1/"+count_str+".jpg", image)     # save frame as JPEG file      
  success,image = vidcap.read()
  #print('Read a new frame: ', success)
  count += 1