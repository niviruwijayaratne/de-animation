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
    feature_mask = (np.zeros_like(reference_frame[:, :, 0])*255).astype(np.uint8)
    feature_mask[y_coords, x_coords] = 1
    return feature_mask

"""
Calculates the average pixel RGB value and saves the average image
:param input:   string          video path
:param output:  string          output image path
:return: numpy.ndarray lxwx3
"""
def get_average_image(input_path, output_path):
    vid = cv2.VideoCapture(input_path)
    success, frame = vid.read()
    average = np.zeros(frame.shape)
    num_frames = 0
    while success:
        num_frames += 1
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) / 255.0
        average += frame.astype(np.float64)
        success, frame = vid.read()
    average = average / num_frames
    path = os.path.join(output_path, input_path.split("/")[-1].split("_")[0] + "_average.jpg")
    cv2.imwrite(output_path, average[:, :, [2,1,0]] * 255.0)
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
    var = np.ma.masked_array(var_avg, mask=np.dstack([m, m, m]))
    # var = np.logical_and(var_avg, np.dstack([m, m, m]))
    return num_frames, np.mean(var)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", required=True, default="./config.yaml",
        help="Path to config file")

    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    output_dir = os.path.join(os.getcwd(), config['ioParams']['outputDir'], config['ioParams']['inputPath'].split("/")[-1].split(".")[0])
    print("OUTPUT PATH", output_dir)
    input_path = os.path.join(output_dir, config['ioParams']['inputPath'].split("/")[-1].split(".")[0] + \
                    "_floating_texture_mapped." + config['ioParams']['inputPath'].split("/")[-1].split(".")[-1])
    print("INPUT PATH", input_path)

    original_input = os.path.join(os.getcwd(), config['ioParams']['inputPath'][2:])
    deanimated_input = os.path.join(output_dir, config['ioParams']['inputPath'].split("/")[-1].split(".")[0] + \
                    "_floating_texture_mapped." + config['ioParams']['inputPath'].split("/")[-1].split(".")[-1])

    original_output = os.path.join(output_dir, original_input.split("/")[-1].split("_")[0] + "_og_average.jpg")
    deanimated_output = os.path.join(output_dir, deanimated_input.split("/")[-1].split("_")[0] + "_de_average.jpg")

    mask = get_mask(output_dir).astype(bool)
    og_average = get_average_image(original_input, original_output)
    de_average = get_average_image(deanimated_input, deanimated_output)

    # print(og_average.shape)
    # print(get_mask(output_dir).shape)
    # mask = np.zeros(og_average.shape, dtype=bool)
    num_frames, og_var = get_variances(original_input, og_average, mask)
    num_frames, de_var = get_variances(deanimated_input, de_average, mask)
    print("# of Frames:", num_frames)
    print("Original Video RGB Variance:", og_var)
    print("Deanimated Video RGB Variance:", de_var)

if __name__ == '__main__':
    main()
