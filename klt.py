import cv2
import numpy as np
import argparse
import yaml
import os

class Tracker():
    def __init__(self, config):
        self.inputPath = config['ioParams']['inputPath']
        self.saveIntermediate = config['ioParams']['intermediateResults']
        self.outdir = os.path.join(os.getcwd(), 'results') if self.saveIntermediate else None
        self.outputPath = os.path.join(self.outdir, config['ioParams']['outputPath']) if self.outdir else None
        self.featureParams = config['featureParams']
        self.trackingParams = config['trackingParams']

        
        if self.outdir and not os.path.exists(self.outdir):
            os.mkdir(self.outdir)


    def draw_corners(self, img, corners):
        corners = np.int0(corners)
        for i in corners:
            x, y = i.ravel()
            cv2.circle(img, (x, y), 2, 255, 10)
        cv2.imwrite(os.path.join(self.outdir, 'features.jpg'), img)
        
    def track_features(self):
        vid = cv2.VideoCapture(self.inputPath)
        ret, start_frame = vid.read()
        if self.saveIntermediate:
            writer = cv2.VideoWriter(self.outputPath ,cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (start_frame.shape[1], start_frame.shape[0]))
        else:
            writer = None
        start_gray = cv2.cvtColor(start_frame, cv2.COLOR_BGR2GRAY)
        corners = cv2.goodFeaturesToTrack(start_gray, **self.featureParams)
        if self.saveIntermediate:
            self.draw_corners(start_frame, corners)

        color = np.random.randint(0, 255, (self.featureParams['maxCorners'], 3))
        mask = np.zeros_like(start_frame)
        old_corners = corners
        while True:
            ret, frame = vid.read() 
            if frame is None:
                break

            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            new_corners, exists, error = cv2.calcOpticalFlowPyrLK(start_gray, frame_gray, old_corners, None, **self.trackingParams)
            good_new_corners = new_corners[exists == 1]
            good_old_corners = old_corners[exists==1]
            for i, (new, old) in enumerate(zip(good_new_corners, good_old_corners)):
                a, b = new.ravel()
                c, d = old.ravel()
                mask = cv2.line(mask, (a,b), (c, d), color[i].tolist(), 5)
                frame = cv2.circle(frame, (a,b), 2, color[i].tolist(), 5)
            img = cv2.add(frame, mask)
            if writer:
                writer.write(img)
            cv2.imshow('frame',img)
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break

            start_gray = frame_gray.copy()
            old_corners = good_new_corners.reshape(-1, 1, 2)
        vid.release()
        if writer:
            writer.release()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", required=True, default="./config.yaml", 
        help="Path to config file")
    
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    tracker = Tracker(config)
    tracker.track_features()




if __name__ == '__main__':
    main()
    
