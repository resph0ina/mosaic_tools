import cv2, os, sys
import numpy as np
from utils import *

def test1():
    img = cv2.imread("e:/1.jpg")
    for i in range(1,5):
        cv2.imshow("image", img)
        cv2.waitKey(0)
        mask = np.zeros((img.shape[0],img.shape[1]), dtype=np.uint8)
        cv2.circle(mask, (i*300, i*300), 300, 255, -1, 8, 0)
        add_mosaic_mask(img, mask, i*5)

def test2():
    rootd='./images/'
    fs=os.listdir(rootd)
    for f in fs:
      img = cv2.imread(rootd+f)
      edges = auto_canny(img)
      w = img.shape[1]
      h = img.shape[0]
      s = min(w,h)
      for _ in xrange(3):
        rw = int(np.random.uniform(s/8, s/3))
        rh = int(np.random.uniform(0.8, 1.2) * rw)
        rx = int(np.random.uniform(0, w-rw))
        ry = int(np.random.uniform(0, h-rh))
        if float(np.sum(edges[ry:(ry+rh), rx:(rx+rw)] != 0)) / (rw*rh) > 0.1:
            add_mosaic_rect(img, (rx,ry), (rx+rw, ry+rh))
            cv2.rectangle(img, (rx,ry), (rx+rw, ry+rh), color=(0,255,0))
      cv2.imshow('edge',edges)
      cv2.imshow('im',img)
      cv2.waitKey(0)

if __name__ == '__main__':
    test1()
  
