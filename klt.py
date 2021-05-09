import cv2
import numpy as np
import argparse
import os
import sys

class Tracker():
    def __init__(self, config, mode = "anchor"):
        self.mode = mode            
        if not os.path.exists(os.path.join(os.getcwd(), config['ioParams']['outputDir'])):
            os.mkdir(os.path.join(os.getcwd(), config['ioParams']['outputDir']))
        if not os.path.exists(os.path.join(os.getcwd(), config['ioParams']['outputDir'], config['ioParams']['inputPath'].split("/")[-1].split(".")[0])):
            os.mkdir(os.path.join(os.getcwd(), config['ioParams']['outputDir'], config['ioParams']['inputPath'].split("/")[-1].split(".")[0]))
        
        self.outdir = os.path.join(os.getcwd(), config['ioParams']['outputDir'], config['ioParams']['inputPath'].split("/")[-1].split(".")[0])
        self.inputPath = {"anchor": config['ioParams']['inputPath'],
                                    "floating": os.path.join(self.outdir, config['ioParams']['inputPath'].split("/")[-1].split(".")[0] \
                                        + "_anchor_texture_mapped." + config['ioParams']['inputPath'].split("/")[-1].split(".")[-1])}

        self.featureParams = config['featureParams']
        self.trackingParams = config['trackingParams']
        self.meshSize = config['meshSize']
        self.num_frames = config['ioParams']['maxFrames']
        self.reference_frame = None
        self.feature_table = {"anchor": None, "floating": None}
        self.vertices = None
        self.quads = None
        self.quad_dict = {}
        self.solved_quad_indices = {"anchor": None, "floating": None}
        self.weights_and_verts = {"anchor": None, "floating": None}
        self.feature_mask = None
        
    def draw_corners(self, img: np.ndarray, corners: np.ndarray):
        '''
        Returns frame with initial features drawn.
        '''
        corners = np.int0(corners)
        for i in corners:
            x, y = i.ravel()
            cv2.circle(img, (x, y), 2, 255, 10)
        cv2.imshow('corners', img)
        cv2.waitKey(0)
         
    def get_de_animate_mask(self):
        '''
        Returns mask of user defined de-animation strokes.
        '''

        y_coords, x_coords = np.load(os.path.join(self.outdir, "y_coords.npy")), np.load(os.path.join(self.outdir, "x_coords.npy"))
        y_coords = y_coords.reshape(-1, 1)
        x_coords = x_coords.reshape(-1, 1)
        stroke_coords = np.hstack([y_coords, x_coords])
        self.feature_mask = np.zeros_like(self.reference_frame[:, :, 0]).astype(np.uint8)
        self.feature_mask[y_coords, x_coords] = 255

    def track_features(self):
        '''
        Tracks and constructs table of features, K(s, t) as defined in Section 5 where s
        indexes over features and t indexes over frames.
        '''
        vid = cv2.VideoCapture(self.inputPath[self.mode])
        ret, start_frame = vid.read()
        self.reference_frame = start_frame
        self.get_de_animate_mask()     
        start_gray = cv2.cvtColor(start_frame, cv2.COLOR_BGR2GRAY)
        corners = cv2.goodFeaturesToTrack(start_gray, **self.featureParams, mask = self.feature_mask)
        old_corners = corners
        frame_count = 0
        feature_tracks = {}
        while True:
            ret, frame = vid.read() 
            if not ret:
                break
            feature_tracks[frame_count] = np.zeros((corners.shape[0], 2))
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            new_corners, exists, error = cv2.calcOpticalFlowPyrLK(start_gray, frame_gray, old_corners, None, **self.trackingParams)
            feature_corners = np.flip(old_corners, axis=2).reshape(-1, 2)
            feature_tracks[frame_count] = feature_corners
            good_new_corners = new_corners
            good_new_corners[~exists.all(axis=1)] = None
            good_old_corners = old_corners
            good_old_corners[~exists.all(axis=1)] = None

            start_gray = frame_gray.copy()
            old_corners = good_new_corners.reshape(-1, 1, 2)

            if self.num_frames > 0:
                if frame_count > self.num_frames - 2:
                    break
            frame_count += 1

        vid.release()
        self.feature_table[self.mode] = np.ones((feature_tracks[list(feature_tracks.keys())[0]].shape[0], max(feature_tracks.keys()) + 1, 2))
        for t in sorted(feature_tracks.keys()):
            for s in range(self.feature_table[self.mode].shape[0]):
                self.feature_table[self.mode][s, t] = np.squeeze(feature_tracks[t][s])

    def get_mesh(self):
        '''
        Constructs 65 x 33 x 2 numpy array of all quad vertices ((y, x) pairs for easy indexing)
        in a frame (raster order). Also constructs 64*32 x 4 x 2 ndarray where each row indexes a 
        mesh quad (raster order). Each row gives a 4x2 vector of the vertices that make up that quad 
        ordered by top left, top right, bottom left, and bottom right vertices.
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
        
        self.quads = np.zeros((64*32, 4,  2))
        count_steps = np.arange(0, (64*32), 1)
        y_steps = np.arange(0, self.vertices.shape[0] - 1, 1).reshape(-1, 1)
        y_steps = np.hstack([y_steps, y_steps + 1]).reshape(-1, 2)
        x_steps = np.arange(0, self.vertices.shape[1] - 1, 1).reshape(-1, 1)
        x_steps = np.hstack([x_steps, x_steps + 1]).reshape(-1, 2)
        x_steps = np.hstack([x_steps, x_steps]).reshape(-1, 2, 2)
        y_steps_final = np.repeat(y_steps, x_steps.shape[0], axis = 0).reshape(-1, 2, 1)
        x_steps_final = np.tile(x_steps, (y_steps.shape[0], 1, 1))

        self.quads[count_steps] = np.squeeze(self.vertices[y_steps_final, x_steps_final]).reshape(-1, 4, 2)

    def construct_quad_dict(self):
        '''
        Returns dict()
        key: tuple(y, x pair)
        value: int that gives index in quads to the quad that key is in the top left corner
        Constructs dictionary where a key is a quad vertex and its corresponding value is an 
        index in self.quads that gives the quad of which the vertex is the top left vertex.

        '''
        for row in self.vertices:
            for pair in row:
                for i, quad in enumerate(self.quads):
                    eq = (pair == quad).all(axis=1)
                    if eq.any():
                        if int(np.where(eq)[0]) != 0:
                            continue
                        elif tuple(pair) not in self.quad_dict:
                            self.quad_dict[tuple(pair)] = 0
                            self.quad_dict[tuple(pair)] = i
                        else:
                            self.quad_dict[tuple(pair)] = i
                                 

    def search_quads(self): 
        '''
        Returns # of frames x # of features ndarray where each row gives the list of quads that 
        include all feature points in that frame and each index in that row corresponds to the quad 
        that contains the feature point with the same index in self.feature_table.
        '''
        features = self.feature_table[self.mode].transpose(1, 0, 2) #frame_number x feature number (y, x) coords
        steps = np.array([self.reference_frame.shape[0]/64, self.reference_frame.shape[1]/32])
        closest = np.floor(features/steps)
        top_left = np.multiply(closest, steps)
        self.solved_quad_indices[self.mode] = np.zeros_like(features[:, :, 0])
        for i, row in enumerate(top_left):
            for j, point in enumerate(row):
                if tuple(point) in self.quad_dict.keys():
                    self.solved_quad_indices[self.mode][i, j] = self.quad_dict[tuple(point)]

    def get_weights(self):
        '''
        Returns # of frames x # of features x 2 x 4 where each index in a row corresponds to a 
        2x4 array where the first element is a 1 x 4 list that contains weights, w and the second 
        element contains indices in self.vertices, that when indexed, give the vertices of the quad
        that encloses that feature. Enables each feature to be represented as the bilinear interpolation
        of the vertices of the quad that encloses it.

        '''
        features = self.feature_table[self.mode].transpose(1, 0, 2)
        vertices = self.vertices.reshape(-1, 2)
        frame_adder = 0
        self.weights_and_verts[self.mode] = np.zeros((features.shape[0], features.shape[1], 2, 4))
        for i, row in enumerate(self.solved_quad_indices[self.mode]):
            frame_adder = i*len(vertices)
            for j, quad_idx in enumerate(row):
                if quad_idx < 0:
                    continue
                else:
                    quad_idx = int(quad_idx)
                quad_vertices = self.quads[quad_idx]
                feature_point = features[i, j]
                weights = self.get_quad_weights(feature_point, quad_vertices)
                solved_vertices = []
                for vertex in quad_vertices:
                    solved_vertices.append(int(np.where((vertices == vertex).all(axis=1))[0]) + frame_adder)
                self.weights_and_verts[self.mode][i, j, :, :] = np.array([weights, solved_vertices])
                
    def get_quad_weights(self, feature_point, quad_vertices):
        '''
        Helper function to calculate the bilinear interpolation weights of each vertex
        in the quad that encloses a feature point.
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
        '''
        Function that runs full tracking pipeline.
        '''
        if self.mode == "anchor":
            print("Anchor Mode")
            self.track_features()
            self.get_mesh()
            self.construct_quad_dict()
            self.search_quads()
            self.get_weights()
        else:
            print("Floating Mode")
            self.track_features()
            self.search_quads()
            self.get_weights()
    