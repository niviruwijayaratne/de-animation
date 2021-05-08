import cv2
import numpy as np
import matplotlib.pyplot as plt

"""
Calculates the average pixel RGB value and saves the average image
:param input:   string          video path
:param output:  string          output image path
:return: numpy.ndarray lxwx3
"""
def get_average_image(input, output):
    vid = cv2.VideoCapture(input)
    success, frame = vid.read()
    average = np.zeros(frame.shape)
    num_frames = 0
    while success:
        num_frames += 1
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) / 255.0
        average += frame.astype(np.float64)
        success, frame = vid.read()
    average = average / num_frames
    cv2.imwrite(output, average[:, :, [2,1,0]] * 255.0)
    return average

"""
Returns the mean RGB pixel variances, computed within the green strokes across all frames of the output video
:param input:   string          video path
:param average: numpy.ndarray   float64 array with the average RGB pixel values
:param m:       numpy.ndarray   boolean array where the green strokes are False and everything else is True
:return: number of frames, variance
(1080, 1920, 3)
"""
def get_variances(input, average, m):
    vid = cv2.VideoCapture(input)
    success, frame = vid.read()
    print(frame.shape)
    var_avg = np.zeros(frame.shape)
    num_frames = 0
    while success:
        num_frames += 1
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) / 255.0
        var = np.sqrt(np.abs(frame - average))
        var_avg += var
        success, frame = vid.read()
    var_avg = var_avg / num_frames
    var = np.ma.masked_array(var_avg, mask=m)
    return num_frames, np.mean(var)

input_path = "inputs/beer.mp4"
output_path = "results/beer_average.png"
average = get_average_image(input_path, output_path)
mask = np.zeros((1080, 1920, 3), dtype=bool)
num_frames, var = get_variances(input_path, average, mask)
print("# of Frames:", num_frames)
print("Input:", var)
print(var)
