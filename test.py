# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 17:13:03 2019

@author: ECO-EMS
"""
import random
from sklearn.model_selection import train_test_split
import argparse
import os
import glob
import tensorflow as tf
import cv2
import numpy as np
from model import *
import sys

def create_y(filename):
    img = cv2.imread(filename, cv2.IMREAD_COLOR)
    (h, w) = img.shape[:2]

    rand_y_shift = random.randint(-int(h*0.05/2), int(h*0.05/2))
    rand_x_shift = random.randint(-int(w*0.05/2), int(w*0.05/2))

    center = (w / 2 + rand_x_shift, h / 2+rand_y_shift)
    angle_rnd = random.randint(0, 360)
    M = cv2.getRotationMatrix2D(center, angle_rnd, 1.0)
    rotated_random = cv2.warpAffine(img, M, (h, w))
    print(filename[:-4]+"-"+str(angle_rnd)+".jpg")
    cv2.imwrite( filename[:-4]+"-"+str(angle_rnd)+".jpg",rotated_random)
    

create_y(sys.argv[1])