import cv2
vidcap = cv2.VideoCapture('[HIGHLIGHT] Bali United FC vs Dewa United FC _ Goal Skill Save.mp4')
success,image = vidcap.read()
print(success)
count = 0
while success:
    cv2.imwrite("../static/results/psis/frame%d.jpg" % count, image)
    success,image = vidcap.read()
    print('Read a new frame: ', success)
    count += 10