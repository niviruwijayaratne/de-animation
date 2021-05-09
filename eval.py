import cv2
import numpy as np
import os
import argparse
import yaml 

def get_mask(outdir):
    '''
    Returns binary mask for de-animation strokes.
    '''
    y_coords, x_coords = np.load(os.path.join(outdir, "y_coords.npy")), np.load(os.path.join(outdir, "x_coords.npy"))
    y_coords = y_coords.reshape(-1, 1)
    x_coords = x_coords.reshape(-1, 1)
    stroke_coords = np.hstack([y_coords, x_coords])
    reference_frame = cv2.imread(os.path.join(outdir, 'reference_frame_anchor.jpg'))
    feature_mask = (np.ones_like(reference_frame[:, :, 0])*255).astype(np.uint8)
    feature_mask[y_coords, x_coords] = 0
    return feature_mask

"""
Calculates the average pixel RGB value and saves the average image
:param input:   string          video path
:param output:  string          output image path
:return: numpy.ndarray lxwx3
"""
def get_average_image(inputVid, output_dir):
    vid = cv2.VideoCapture(inputVid)
    success, frame = vid.read()
    average = np.zeros(frame.shape)
    num_frames = 0
    while success:
        num_frames += 1
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) / 255.0
        average += frame.astype(np.float64)
        success, frame = vid.read()
    average = average / num_frames
    cv2.imwrite(os.path.join(output_dir, inputVid.split("/")[-1].split("_")[0] + "_average.jpg"), average[:, :, [2,1,0]] * 255.0)
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
    var_avg = np.zeros(frame.shape)
    num_frames = 0
    while success:
        num_frames += 1
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) / 255.0
        var = np.sqrt(np.abs(frame - average))
        var_avg += var
        success, frame = vid.read()
    var_avg = var_avg / num_frames
    # var = np.ma.masked_array(var_avg, mask=m)
    var = np.logical_and(var_avg, np.dstack([m, m, m]))
    return num_frames, np.mean(var)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", required=True, default="./config.yaml", 
        help="Path to config file")
    
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    output_path = os.path.join(os.getcwd(), config['ioParams']['outputDir'], config['ioParams']['inputPath'].split("/")[-1].split(".")[0])
    input_path = os.path.join(output_path, config['ioParams']['inputPath'].split("/")[-1].split(".")[0] + \
                    "_floating_texture_mapped." + config['ioParams']['inputPath'].split("/")[-1].split(".")[-1])
    mask = get_mask(output_path).astype(bool)
    average = get_average_image(input_path, output_path)
    # mask = np.zeros((1080, 1920, 3), dtype=bool)
    num_frames, var = get_variances(input_path, average, mask)    
    print("# of Frames:", num_frames)
    print("Input RGB Variance:", var)

if __name__ == '__main__':
    main()