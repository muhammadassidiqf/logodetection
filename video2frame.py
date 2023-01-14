import cv2
vidcap = cv2.VideoCapture('videplayback.mp4')
success,image = vidcap.read()
print(success)
count = 0
while success:
    cv2.imwrite("../static/results/psis/frame%d.jpg" % count, image)
    success,image = vidcap.read()
    print('Read a new frame: ', success)
    count += 10