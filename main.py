import cv2
from time import sleep
import numpy as np

draw = True

class Person:

    def __init__(self, x, y):
        self.x = x
        self.y = y



img = np.zeros((512,512,3), np.uint8)
cv2.line(img,(0,0),(511,511),(255,0,0),5)
cv2.circle(img,(50,50),18,(0,255,0),-1)
cv2.namedWindow('image')


for i in range(1000):
    cv2.imshow('image',img)
    cv2.circle(img,(int(i/2),250),18,(0,255,0),-1)
    cv2.waitKey(33)


cv2.waitKey(0)
cv2.destroyAllWindows()
