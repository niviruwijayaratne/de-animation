import cv2
import numpy as np
import argparse
import yaml
import os
import sys

class Tracker():
    def __init__(self, config):
        self.inputPath = config['ioParams']['inputPath']
        self.saveIntermediate = config['ioParams']['intermediateResults']
        self.outdir = os.path.join(os.getcwd(), 'results') if self.saveIntermediate else None
        self.outputPath = os.path.join(self.outdir, config['ioParams']['outputPath']) if self.outdir else None
        self.featureParams = config['featureParams']
        self.trackingParams = config['trackingParams']
        self.meshSize = config['meshSize']
        self.reference_frame = None

        self.feature_table = None
        self.vertices = None
        self.quads = None
        self.quad_dict = {}
        self.solved_quad_indices = None
        self.weights_and_verts = None
        self.solved_quad_indices = []

        if self.outdir and not os.path.exists(self.outdir):
            os.mkdir(self.outdir)
        
    def draw_corners(self, img: np.ndarray, corners: np.ndarray):
        corners = np.int0(corners)
        for i in corners:
            x, y = i.ravel()
            cv2.circle(img, (x, y), 2, 255, 10)
        cv2.imwrite(os.path.join(self.outdir, 'features.jpg'), img)
    
    def track_features(self):
        '''
        Solves for s, t table of features where s indexes over different features and t indexes over frames
        '''

        vid = cv2.VideoCapture(self.inputPath)
        ret, start_frame = vid.read()
        self.reference_frame = start_frame
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
        frame_count = 0
        feature_tracks = {}
        while True:
            ret, frame = vid.read() 
            if frame is None:
                break
            feature_tracks[frame_count] = np.zeros((corners.shape[0], 2))
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            new_corners, exists, error = cv2.calcOpticalFlowPyrLK(start_gray, frame_gray, old_corners, None, **self.trackingParams)
            feature_tracks[frame_count] = np.flip(old_corners, axis=2)
            
            good_new_corners = new_corners
            good_new_corners[~exists.all(axis=1)] = [0, 0]
            good_old_corners = old_corners
            good_old_corners[~exists.all(axis=1)] = [0, 0]

            for i, (new, old) in enumerate(zip(good_new_corners, good_old_corners)):
                a, b = new.ravel()
                c, d = old.ravel()
                mask = cv2.line(mask, (a,b), (c, d), color[i].tolist(), 5)
                frame = cv2.circle(frame, (a,b), 2, color[i].tolist(), 5)
            img = cv2.add(frame, mask)
            # if writer:
            #     writer.write(img)
            # cv2.imshow('frame',img)
            # k = cv2.waitKey(30) & 0xff
            # if k == 27:
            #     break
            start_gray = frame_gray.copy()
            old_corners = good_new_corners.reshape(-1, 1, 2)
            if frame_count >= 5:
                break
            frame_count += 1
            
        if writer:
            writer.release()

        self.construct_table(feature_tracks)

    def construct_table(self, feature_tracks) -> np.ndarray:
        self.feature_table = np.zeros((self.featureParams['maxCorners'], max(feature_tracks.keys()) + 1, 2))
        count = 0
        for t in feature_tracks.keys():
            for s in range(self.featureParams['maxCorners']):
                self.feature_table[s, t] = np.squeeze(feature_tracks[t][s])
    def get_mesh(self):
        '''
        Returns: 64*32 x 4 x 2 ndarray where each row indexes a mesh quad (raster order)
        and gives the 4x2 vector of the vertices that define that quad; order = 
        top left, top right, bottom left, bottom right
        '''
        quads = np.zeros((self.meshSize[0]*self.meshSize[1], 4, 2))
        h, w = self.reference_frame.shape[0], self.reference_frame.shape[1]
        y_step, x_step = h/self.meshSize[0], w/self.meshSize[1]
        x_coords = np.arange(0, w + 1, x_step)
        y_coords = np.arange(0, h + 1, y_step)
        assert(len(x_coords) == self.meshSize[1] + 1)
        assert(len(y_coords) == self.meshSize[0] + 1)
        x_grid, y_grid = np.meshgrid(x_coords, y_coords)
        
        self.vertices = np.zeros((65, 33, 2))
        for i, y in enumerate(y_grid):
            self.vertices[i, :, :] = np.array(list(zip(y, x_grid[0])))

        x_steps1 = np.arange(0, self.vertices.shape[1] + 1, 2)
        x_steps2 = np.arange(1, self.vertices.shape[1] + 2, 2)
        y_steps1 = np.arange(0, self.vertices.shape[0] + 1, 2)
        y_steps2 = np.arange(1, self.vertices.shape[0] + 2, 2)

        self.quads = np.zeros((64*32, 4,  2))
        count = 0
        for j in range(len(y_steps1) - 1):
            for i in range(len(x_steps1) - 1):
                pairs = np.squeeze(self.vertices[y_steps1[j]:y_steps1[j + 1], x_steps1[i]:x_steps1[i+1]])
                self.quads[count] = np.array([pairs[0, 0], pairs[0, 1], pairs[1, 0], pairs[1, 1]])
                count += 1
                pairs = np.squeeze(self.vertices[y_steps1[j]:y_steps1[j + 1], x_steps2[i]:x_steps2[i+1]])
                self.quads[count] = np.array([pairs[0, 0], pairs[0, 1], pairs[1, 0], pairs[1, 1]])
                count += 1
            
            for i in range(len(x_steps1) - 1):
                pairs = np.squeeze(self.vertices[y_steps2[j]:y_steps2[j + 1], x_steps1[i]:x_steps1[i+1]])
                self.quads[count] = np.array([pairs[0, 0], pairs[0, 1], pairs[1, 0], pairs[1, 1]])
                count += 1
                pairs = np.squeeze(self.vertices[y_steps2[j]:y_steps2[j + 1], x_steps2[i]:x_steps2[i+1]])
                self.quads[count] = np.array([pairs[0, 0], pairs[0, 1], pairs[1, 0], pairs[1, 1]])
                count += 1

    def construct_quad_dict(self):
        '''
        Returns dict()
        key: tuple(y, x pair)
        value: int that gives index in quads to the quad that key is in the top left corner
        '''
        for row in self.vertices:
            for pair in row:
                for i, quad in enumerate(self.quads):
                    eq = (pair == quad).all(axis=1)
                    if eq.any():
                        if np.where(eq)[0] != 0:
                            continue
                        else:
                            self.quad_dict[tuple(pair)] = i
        
    def search_quads(self): 
        '''
        Returns # of frames x s ndarray where each row gives the list of quads that include all feature point in that frame
        and each index in that row corresponds to the quad that contains the feature point with the same index in the tracktable
        '''
        features = self.feature_table.transpose(1, 0, 2) #frame_number x feature number (y, x) coords
        steps = np.array([self.reference_frame.shape[0]/64, self.reference_frame.shape[1]/32])
        closest = np.floor(features/steps)
        top_left = np.multiply(closest, steps)
        self.solved_quad_indices = np.zeros_like(features[:, :, 0])
        for i, row in enumerate(top_left):
            for j, point in enumerate(row):
                if tuple(point) in self.quad_dict.keys():
                    self.solved_quad_indices[i, j] = self.quad_dict[tuple(point)]
        self.solved_quad_indices = self.solved_quad_indices.astype(np.int64)

    def get_weights(self):
        '''
        Returns num_frames x num_features x 2x4 where each index in a row corresponds to a 2x4 array where the first row contains the weights 
        and the second row contains the point indices in an array that is a 2-column matrix of self.xy_pairs repeated t times in [tl, tr, bl, br] order
        '''
        features = self.feature_table.transpose(1, 0, 2)
        vertices = self.vertices.reshape(-1, 2)
        frame_adder = 0
        self.weights_and_verts = np.zeros((features.shape[0], features.shape[1], 2, 4))
        for i, row in enumerate(self.solved_quad_indices):
            frame_adder = i*len(vertices)
            for j, quad_idx in enumerate(row):
                quad_vertices = self.quads[quad_idx]
                feature_point = features[i, j]
                weights = self.get_quad_weights(feature_point, quad_vertices)
                solved_vertices = []
                for vertex in quad_vertices:
                    solved_vertices.append(int(np.where((vertices == vertex).all(axis=1))[0]) + frame_adder)
                self.weights_and_verts[i, j, :, :] = np.array([weights, solved_vertices])

    def get_quad_weights(self, feature_point, quad_vertices):
        '''
        Inputs:
        - features: np.ndarray of shape (1x2) (x, y) coordinates of a feature point
        - vertices: vertices of the qaud that encloses that feature, in tl, tr, bl, br order, (4x2) array

        - returns (4x1) array of weights for bilinear interpolation
        '''
        tl, tr, bl, br = quad_vertices
        total_area = (br[1] - tl[1])*(br[0] - tl[0])
        tl_area = (feature_point[1] - tl[1])*(feature_point[0] - tl[0])
        tr_area = (tr[1] - feature_point[1])*(feature_point[0] - tr[0])
        bl_area = (feature_point[1] - bl[1])*(bl[0] - feature_point[0])
        br_area = (br[1] - feature_point[1])*(br[0] - feature_point[0])
 
        tl_weight = br_area/total_area
        tr_weight = bl_area/total_area
        bl_weight = tr_area/total_area
        br_weight = tl_area/total_area

        return [tl_weight, tr_weight, bl_weight, br_weight]
    
    def run(self):
        self.track_features()
        self.get_mesh()
        self.construct_quad_dict()
        self.search_quads()
        self.get_weights()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", required=True, default="./config.yaml", 
        help="Path to config file")
    
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    tracker = Tracker(config)
    tracker.run()
    
if __name__ == '__main__':
    main()
    