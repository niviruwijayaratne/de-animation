import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import yaml
import os
import sys
import time
import threading

def get_average_image(input, output):
    vid = cv2.VideoCapture(input)
    success, frame = vid.read()
    average = np.zeros(frame.shape)
    num_frames = 0
    while success:
        num_frames += 1
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        average += frame.astype(np.float64)
        success, frame = vid.read()
    average = average / num_frames
    cv2.imwrite(output, average)


get_average_image("inputs/beer.mp4", "results/beer_average.png")
