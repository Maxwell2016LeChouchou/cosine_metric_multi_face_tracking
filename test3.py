import cv2
vidcap = cv2.VideoCapture('/home/max/Videos/MV_frame/brunomars.mp4')
success,image = vidcap.read()
count = 1
while success:
  count_str = str(count)
  while len(count_str) < 4:
    count_str = "0" + count_str
  cv2.imwrite("/home/max/Videos/BrunoMars/"+count_str+".jpg", image)     # save frame as JPEG file      
  success,image = vidcap.read()
  #print('Read a new frame: ', success)
  count += 1