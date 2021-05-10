import numpy as np
import scipy as sc
import scipy.sparse.linalg
import cv2
from math import sqrt
from klt import Tracker
import yaml
import argparse
import sys
import time
import os
import tqdm
from user_input import StrokeCanvas


class LeastSquaresSolver:
    def __init__(self, tracker):
        self.tracker = tracker
        self.ka = {"anchor": None, "floating": None}
        self.vertices = None
        self.weights_and_verts = {"anchor": None, "floating": None}
        self.non_solved_verts = {"anchor": None, "floating": None}
        self.processed_quads = None
        self.temporal_weight_mat = {"anchor": None, "floating": None}
        
    def pre_process(self):
        '''
        Pre-processes outputs from tracking to simplify least squares problem.
        '''
        mode = self.tracker.mode
        if mode == "anchor":    
            num_s = self.tracker.feature_table["anchor"].shape[0]
            num_t = self.tracker.feature_table["anchor"].shape[1]
            self.ka[mode] = self.tracker.feature_table[mode].transpose(1, 0, 2)[0].reshape(-1, 2)
            self.ka[mode] = self.tracker.feature_table[mode].transpose(1, 0, 2)[0].reshape(-1, 1)
            self.ka[mode] = np.tile(self.ka[mode], (num_t, 1))
            assert(self.ka[mode].shape == (num_s*num_t*2, 1))

            self.processed_quads = np.zeros((64*32, 4))
            vertices = self.tracker.vertices.reshape(-1, 2)
            for i, quad in enumerate(self.tracker.quads):
                quad_idxs = []
                for vert in quad:
                    quad_idxs.append(np.where((vertices == vert).all(axis=1))[0][0])
                self.processed_quads[i, :] = quad_idxs

            assert(self.processed_quads.shape == (64*32, 4))
            self.processed_quads = np.tile(self.processed_quads, (num_t, 1))
            assert(self.processed_quads.shape == (64*32*num_t, 4))
            for i in range(0, self.processed_quads.shape[0], 64*32):
                self.processed_quads[i:(64*32)*(int(i/2048) + 1), :] += len(vertices)*(int(i/2048))
            self.processed_quads = np.repeat(self.processed_quads, 2, axis=0)
            assert(self.processed_quads.shape == (64*32*2*num_t, 4))
            self.processed_quads *= 2
            self.processed_quads[np.arange(1, self.processed_quads.shape[0] + 1, 2)] += 1
            self.processed_quads = self.processed_quads.astype(np.int64)

            self.vertices = vertices.reshape(-1, 1)
            assert(self.vertices.shape == (65*33*2, 1))
            self.vertices = np.tile(self.vertices, (num_t, 1))
            assert(self.vertices.shape == (65*33*2*num_t, 1))

            self.weights_and_verts[mode] = self.tracker.weights_and_verts[mode].reshape(-1, 2, 4)
            assert(self.weights_and_verts[mode].shape == (num_s*num_t, 2, 4))
            self.weights_and_verts[mode] = np.repeat(self.weights_and_verts[mode], 2, axis=0)
            assert(self.weights_and_verts[mode].shape == (num_s*num_t*2, 2, 4))
            self.weights_and_verts[mode][:, 1, :] *= 2
            x_coords = np.arange(1, self.weights_and_verts[mode].shape[0] + 1, 2).astype(np.int64)
            self.weights_and_verts[mode][x_coords, 1, :] += 1

        else:
            num_s = self.tracker.feature_table["floating"].shape[0]
            num_t = self.tracker.feature_table["floating"].shape[1]
            self.weights_and_verts[mode] = self.tracker.weights_and_verts[mode].transpose(1, 0, 2, 3)
            assert(self.weights_and_verts[mode].shape == (num_s, num_t, 2, 4))
            self.weights_and_verts[mode] = np.hstack([self.weights_and_verts[mode][:, :-1, :, :].reshape(-1, 2, 4), self.weights_and_verts[mode][:, 1:, :, :].reshape(-1, 2, 4)])
            assert(self.weights_and_verts[mode].shape == (num_s*(num_t - 1), 4, 4)) #weights t, verts t, weights t+1, verts t + 1
            self.weights_and_verts[mode] = np.repeat(self.weights_and_verts[mode], 2, axis=0)
            assert(self.weights_and_verts[mode].shape == (num_s*(num_t - 1)*2, 4, 4))
            self.weights_and_verts[mode][:, 1, :] *= 2
            self.weights_and_verts[mode][:, 3, :] *= 2
            x_coords = np.arange(1, self.weights_and_verts[mode].shape[0] + 1, 2).astype(np.int64)
            self.weights_and_verts[mode][x_coords, 1, :] += 1
            self.weights_and_verts[mode][x_coords, 3, :] += 1

        self.non_solved_verts[mode] = []
        vers = np.arange(0, self.vertices.shape[0], 1)
        reshaped_idxs = np.unique(self.weights_and_verts[mode][:, 1, :].reshape(-1, 1), axis=0)
        for v in vers:
            if v not in reshaped_idxs:
                self.non_solved_verts[mode].append(v)
        self.non_solved_verts[mode] = np.array(self.non_solved_verts[mode]).reshape(-1, 1)
        if mode == "floating":
            self.non_solved_verts["floating"] = np.intersect1d(self.non_solved_verts["anchor"], self.non_solved_verts["floating"]).reshape(-1, 1)
        
        self.temporal_weight_mat[mode] = self.temporal_weight_mat[mode].transpose(1, 0).reshape(-1, 1)
        self.temporal_weight_mat[mode] = np.repeat(self.temporal_weight_mat[mode], 2, axis=0)

    def energy_function_lsq(self):
        '''
        Minimize energy functions (3) and (5) using least squares, as outlined in paper.
        Returns (65*33*2* #frames) x 1 array where every 2 indices forms a vertex (y, x) 
        pair (for easy indexing).
        '''
        mode = self.tracker.mode
        num_s_a = self.tracker.feature_table["anchor"].shape[0]
        num_t_a = self.tracker.feature_table["anchor"].shape[1]
        num_s_f = None
        num_t_f = None
        if mode == "floating":
            num_s_f = self.weights_and_verts["floating"].shape[0]
            num_t_f = self.weights_and_verts["floating"].shape[1]

        if mode == "anchor":    
            ea_rows = self.ka["anchor"].shape[0] + self.non_solved_verts["anchor"].shape[0]     
        else:
            ea_rows = self.ka["anchor"].shape[0] + self.non_solved_verts["floating"].shape[0] + self.weights_and_verts["floating"].shape[0]

        es_rows = 64*32*8*2*num_t_a
        ka_final = np.zeros((ea_rows+es_rows, 1))
        ka_final[:self.ka["anchor"].shape[0]] = np.multiply(self.ka["anchor"], np.sqrt(self.temporal_weight_mat["anchor"]))
        A = scipy.sparse.lil_matrix((ea_rows + es_rows, len(self.vertices)))
        
        print('\r' + "Setting up Least Squares...", end = "")
        idxs = np.arange(0, self.weights_and_verts["anchor"].shape[0], 1).reshape(-1, 1)
        vert_idxs = np.squeeze(self.weights_and_verts["anchor"][idxs, 1].astype(np.int64))
        weights = np.squeeze(self.weights_and_verts["anchor"][idxs, 0])
        A[idxs, vert_idxs] = np.multiply(weights, np.sqrt(self.temporal_weight_mat["anchor"]))
        
        # print("Started NSV Equations")
        idxs = np.arange(self.weights_and_verts["anchor"].shape[0], self.non_solved_verts[mode].shape[0] + self.weights_and_verts["anchor"].shape[0], 1).reshape(-1, 1)
        A[idxs, self.non_solved_verts[mode]] = 1
        ka_final[idxs] = self.vertices[self.non_solved_verts[mode]]

        if mode == "floating":
            # print("Starting E_f Equations")
            curr_idx = self.non_solved_verts[mode].shape[0] + self.weights_and_verts["anchor"].shape[0]
            idxs = np.arange(curr_idx, curr_idx + self.weights_and_verts["floating"].shape[0], 1).reshape(-1, 1, 1)
            vert_idxs = np.squeeze(self.weights_and_verts["floating"][idxs - curr_idx, [1, 3]]).reshape(-1, 8)
            weights = np.squeeze(self.weights_and_verts["floating"][idxs - curr_idx, [0, 2]])
            weights[:, 1, :] *= -1
            weights = weights.reshape(-1, 8)
            weights = np.multiply(weights, np.sqrt(self.temporal_weight_mat[mode]))
            idxs = idxs.reshape(-1, 1)
            A[idxs, vert_idxs] = weights

        
        # print("Started E_s Equations")
        ea_offset = ea_rows

        es_idxs_y = (np.arange(0, es_rows, 16)/8).astype(np.int64).reshape(-1, 1)
        es_idxs_x = es_idxs_y + 1
        quads_y = np.squeeze(self.processed_quads[es_idxs_y])
        quads_x = np.squeeze(self.processed_quads[es_idxs_x])
        y1, y2, y3, y4 = quads_y[:, 0].reshape(-1, 1), quads_y[:, 1].reshape(-1, 1), quads_y[:, 2].reshape(-1, 1), quads_y[:, 3].reshape(-1, 1)
        x1, x2, x3, x4 = quads_x[:, 0].reshape(-1, 1), quads_x[:, 1].reshape(-1, 1), quads_x[:, 2].reshape(-1, 1), quads_x[:, 3].reshape(-1, 1)
        
        combos_x = np.array([np.hstack([x1, x2, x4]), np.hstack([x4, x2, x1]), np.hstack([x1, x3, x4]), np.hstack([x4, x3, x1]), np.hstack([x2, x1, x3]), np.hstack([x3, x1, x2]), np.hstack([x2, x4, x3]), np.hstack([x3, x4, x2])]).astype(np.int64)
        combos_y = np.array([np.hstack([y1, y2, y4]), np.hstack([y4, y2, y1]), np.hstack([y1, y3, y4]), np.hstack([y4, y3, y1]), np.hstack([y2, y1, y3]), np.hstack([y3, y1, y2]), np.hstack([y2, y4, y3]), np.hstack([y3, y4, y2])]).astype(np.int64)
        x_idxs = np.hstack([combos_x[:, :, 0].reshape(-1, 1), combos_x[:, :, 1].reshape(-1, 1), combos_y[:, :, 2].reshape(-1, 1), combos_y[:, :, 1].reshape(-1, 1)])
        y_idxs = np.hstack([combos_y[:, :, 0].reshape(-1, 1), combos_y[:, :, 1].reshape(-1, 1), combos_x[:, :, 1].reshape(-1, 1), combos_x[:, :, 2].reshape(-1, 1)])
        y_js = (np.arange(0, es_rows, 2) + ea_offset).reshape(-1, 1)
        x_js = y_js + 1
        A[y_js, y_idxs] = np.array([1, -1, -1, 1])*np.sqrt(4)/np.sqrt(8)
        A[x_js, x_idxs] = np.array([1, -1, -1, 1])*np.sqrt(4)/np.sqrt(8)
        print("Set Up!")

        print('\r' + "Solving for Mesh...", end = "")
        v_prime = scipy.sparse.linalg.lsqr(A.tocsr(), ka_final) # solve w/ csr
        print("Solved!")
        v_prime = v_prime[0]
        z = 0
        for i in range(0, len(v_prime), 2):
            if v_prime[i] <= 0:
                z += 1
            if v_prime[i + 1] <= 0:
                z += 1
            if v_prime[i] > self.tracker.reference_frame.shape[0]:
                z += 1
            if v_prime[i + 1] > self.tracker.reference_frame.shape[1]:
                z += 1
            if np.isnan(v_prime[i]) or v_prime[i] is None:
                z += 1
            if np.isnan(v_prime[i + 1]) or v_prime[i + 1] is None:
                z += 1
        
        print(str(100 - (z/len(v_prime)*100)) + "% Valid Points")
        return v_prime

    def get_temporal_weights(self):
        '''
        Creates array of temporal weights based on temporal weighting function
        l(s, t) as outlined in Liu et. al. Returns # features x # frames array
        where each row corresponds to the temporal weights of a feature and each
        index in that row gives the temporal weight of that feature in that frame
        '''
        mode = self.tracker.mode
        s, t = self.tracker.feature_table[mode].shape[0], self.tracker.feature_table[mode].shape[1]
        if mode == "floating":
            t -= 1
        self.temporal_weight_mat[mode] = np.zeros((s, t))
        T = 15
        eps = 1e-2
        for s_i in range(s):
            time_frame = np.where(self.tracker.feature_table[mode][s_i] != None)
            t_start = time_frame[0][0]
            t_end = time_frame[0][-1]
            for t_i in range(t):
                #check if track
                feature = self.tracker.feature_table[mode][s_i, t_i]
                if (feature != None).all():
                    if ( (t_i >= t_start) and (t_i < t_start + T) ):
                        self.temporal_weight_mat[mode][s_i, t_i] = ((t_i - t_start) / T)
                    elif ( (t_i >= t_start + T) and (t_i <= t_end - T) ):
                        self.temporal_weight_mat[mode][s_i, t_i] = 1
                    else:
                        self.temporal_weight_mat[mode][s_i, t_i] = ((t_end - t_i) / T)

    def draw_rectangles(self, v_prime):
        '''
        Draws warped mesh over input video.
        '''
        vid = cv2.VideoCapture(self.tracker.inputPath[self.tracker.mode])
        ret, frame = vid.read()
        count = 0
        h, w = self.tracker.reference_frame.shape[0], self.tracker.reference_frame.shape[1]
        out_name = self.tracker.inputPath['anchor'].split("/")[-1].split(".")[0] + "_" + self.tracker.mode \
            + "_mesh." + self.tracker.inputPath['anchor'].split("/")[-1].split(".")[-1]                    
        out_path = os.path.join(self.tracker.outdir, out_name)
        writer = cv2.VideoWriter(out_path ,cv2.VideoWriter_fourcc(*'mp4v'), vid.get(cv2.CAP_PROP_FPS), (w, h))
        with tqdm.tqdm(total=int(self.processed_quads.shape[0]/(64*32*2)), file=sys.stdout, desc="Drawing Mesh", ncols = 85) as pbar:
            for i in range(0, self.processed_quads.shape[0], 2):
                y_tl = v_prime[self.processed_quads[i][0]]
                x_tl = v_prime[self.processed_quads[i + 1][0]]

                y_tr = v_prime[self.processed_quads[i][1]]
                x_tr = v_prime[self.processed_quads[i + 1][1]]

                y_bl = v_prime[self.processed_quads[i][2]]
                x_bl = v_prime[self.processed_quads[i + 1][2]]

                y_br = v_prime[self.processed_quads[i][3]]
                x_br = v_prime[self.processed_quads[i + 1][3]]
                
                y_tl = int(np.rint(y_tl))
                x_tl = int(np.rint(x_tl))

                y_tr = int(np.rint(y_tr))
                x_tr = int(np.rint(x_tr))

                y_bl = int(np.rint(y_bl))
                x_bl = int(np.rint(x_bl))

                y_br = int(np.rint(y_br))
                x_br = int(np.rint(x_br))

                cv2.line(frame, (x_tl, y_tl), (x_tr, y_tr), (255, 0, 0), 1)
                cv2.line(frame, (x_tr, y_tr), (x_br, y_br), (255, 0, 0), 1)
                cv2.line(frame, (x_br, y_br), (x_bl, y_bl), (255, 0, 0), 1)
                cv2.line(frame, (x_bl, y_bl), (x_tl, y_tl), (255, 0, 0), 1)

                y_tl_o = self.vertices[self.processed_quads[i][0]]
                x_tl_o = self.vertices[self.processed_quads[i + 1][0]]
                
                y_tr_o = self.vertices[self.processed_quads[i][1]]
                x_tr_o = self.vertices[self.processed_quads[i + 1][1]]

                y_bl_o = self.vertices[self.processed_quads[i][2]]
                x_bl_o = self.vertices[self.processed_quads[i + 1][2]]

                y_br_o = self.vertices[self.processed_quads[i][3]]
                x_br_o = self.vertices[self.processed_quads[i + 1][3]]

                if (i + 2) % (64*32*2) == 0:
                    pbar.update(1)
                    writer.write(frame)
                    ret, frame = vid.read()
                    # f = cv2.imread('inputs/reference_frame.jpg')
        vid.release()
        writer.release()

    def texture_map(self, v_prime):
        '''
        Texture maps input video to output mesh.
        '''
        vid = cv2.VideoCapture(self.tracker.inputPath[self.tracker.mode])
        ret, frame = vid.read() 
        out_name = self.tracker.inputPath['anchor'].split("/")[-1].split(".")[0] + "_" + self.tracker.mode + "_texture_mapped." \
                    + self.tracker.inputPath['anchor'].split("/")[-1].split(".")[-1]
        out_path = os.path.join(self.tracker.outdir, out_name)
        writer = cv2.VideoWriter(out_path ,cv2.VideoWriter_fourcc(*'mp4v'), vid.get(cv2.CAP_PROP_FPS), (frame.shape[1], frame.shape[0]))
        warped_frame = np.zeros_like(frame)

        counter = 0
        with tqdm.tqdm(total=int(self.processed_quads.shape[0]/(64*32*2)), file=sys.stdout, desc="Texture Mapping", ncols = 85) as pbar:
            for i in range(0, self.processed_quads.shape[0], 2):
                y_tl = v_prime[self.processed_quads[i][0]]
                x_tl = v_prime[self.processed_quads[i + 1][0]]

                y_tr = v_prime[self.processed_quads[i][1]]
                x_tr = v_prime[self.processed_quads[i + 1][1]]

                y_bl = v_prime[self.processed_quads[i][2]]
                x_bl = v_prime[self.processed_quads[i + 1][2]]

                y_br = v_prime[self.processed_quads[i][3]]
                x_br = v_prime[self.processed_quads[i + 1][3]]

                y_tl_o = self.vertices[self.processed_quads[i][0]]
                x_tl_o = self.vertices[self.processed_quads[i + 1][0]]
                
                y_tr_o = self.vertices[self.processed_quads[i][1]]
                x_tr_o = self.vertices[self.processed_quads[i + 1][1]]

                y_bl_o = self.vertices[self.processed_quads[i][2]]
                x_bl_o = self.vertices[self.processed_quads[i + 1][2]]

                y_br_o = self.vertices[self.processed_quads[i][3]]
                x_br_o = self.vertices[self.processed_quads[i + 1][3]]

                dst = np.array([[y_tl, x_tl], [y_tr, x_tr], [y_bl, x_bl], [y_br, x_br]])
                src = np.array([[y_tl_o, x_tl_o], [y_tr_o, x_tr_o], [y_bl_o, x_bl_o], [y_br_o, x_br_o]])
                transform = self.get_transform(src, dst)
                # bounds = self.get_pixels(dst)
                for y in range(int(np.rint(y_tl)), int(np.rint(y_bl))):
                    for x in range(int(np.rint(x_tl)), int(np.rint(x_tr))):
                        transformed_point = np.dot(transform, np.array([y, x, 1]))
                        transformed_point /= transformed_point[-1]
                        transformed_point[np.where(transformed_point < 0)] = 0
                        transformed_point[0] = min(transformed_point[0], frame.shape[0] - 1)
                        transformed_point[1] = min(transformed_point[1], frame.shape[1] - 1)
                        intensity = frame[int(transformed_point[0]), int(transformed_point[1]), :] 
                        warped_frame[y, x, :] = intensity
                if (i + 2) % (64*32*2) == 0:
                    # print("Writing Frame: ", counter)
                    pbar.update(1)
                    writer.write(warped_frame)
                    warped_frame = np.zeros_like(frame)
                    counter += 1
                    ret, frame = vid.read()
                    if not ret:
                        writer.release()
                        vid.release()
                        return
        writer.release()
        vid.release()

    def get_pixels(self, points):
        '''
        Scan conversion for correctly traversing quad pixels.
        '''
        points = np.squeeze(points)
        tri1 = np.array([points[0], points[1], points[2]])
        tri2 = np.array([points[1], points[2], points[3]])

        pairs1 = np.array([[tri1[1], tri1[0]], [tri1[1], tri1[2]]])
        pairs2 = np.array([[tri1[0], tri1[1]], [tri1[0], tri1[2]]])
        edges = [pairs1, pairs2]
        bounds = {}
        for edge in edges:
            dxl = edge[0, 0, 0] - edge[0, 1, 0]
            dyl = edge[0, 0, 1] - edge[0, 1, 1]
            slope_l = dyl/dxl
            intercept_l = edge[0, 0, 1] - edge[0, 0, 0]*slope_l
            l_val = lambda y: (y - intercept_l)/slope_l

            dxr = edge[1, 0, 0] - edge[1, 1, 0]
            dyr = edge[1, 0, 1] - edge[1, 1, 1]
            slope_r = dyr/dxr
            intercept_r = edge[1, 0, 1] - edge[1, 0, 0]*slope_r
            r_val = lambda y: (y - intercept_r)/slope_r

            max_y = -float('inf')
            min_y = float('inf')
            for pair in edge:
                for point in pair:
                    if point[0] < min_y:
                        min_y = point[0]
                    if point[0] > max_y:
                        max_y = point[0]
            
            for y in range(int(np.ceil(min_y)), int(np.floor(max_y))):
                curr_x = l_val(y)
                if y in bounds:
                    bounds[y].append(curr_x)
                else:
                    bounds[y] = [curr_x]
                while curr_x <= r_val(y):
                    curr_x += np.abs(slope_l)
                    # break
                bounds[y].append(curr_x)

        for k in bounds.keys():
            bounds[k] = [int(max(min(bounds[k]), 0)), int(max(bounds[k]))]
        
        return bounds


    def get_transform(self, src, dst):
        '''
        Returns transformation matrix that transforms warped quad back to original.
        '''
        src = np.squeeze(src)
        dst = np.squeeze(dst)
        src = np.hstack([src, np.ones((src.shape[0], 1))])
        dst = np.hstack([dst, np.ones((dst.shape[0], 1))])

        src_consts = np.dot(np.linalg.inv(src.T[:, :-1]), src.T[:, -1]).reshape(1, -1)
        A = src.T[:, :-1]*src_consts

        dst_consts = np.dot(np.linalg.inv(dst.T[:, :-1]), dst.T[:, -1]).reshape(1, -1)
        B = dst.T[:, :-1]*dst_consts

        T = np.dot(B, np.linalg.inv(A))
        T_inv = np.linalg.inv(T)

        return T_inv

    # def get_transform(self, src, dst):
    #     '''
    #     Helper function to calculate quad-to-quad transforms for texture mapping.
    #     '''
    #     A = np.zeros((src.shape[0]*2, 9))
    #     for i in range(0, A.shape[0], 2):
    #         u, v = np.squeeze(src[int(i/2)])
    #         u_, v_ = np.squeeze(dst[int(i/2)])

    #         A[i] = [-u, -v, -1, 0, 0, 0, u*u_, v*u_, u_]
    #         A[i + 1] = [0, 0, 0, -u, -v, -1, u*v_, v*v_, v_]

    #     U,S,V_t = scipy.linalg.svd(A)
    #     T = V_t[-1]
    #     T = T.reshape((3,3))
    #     T_inv = np.linalg.inv(T)
    #     return T_inv
        
    def run(self):
        '''
        Runs entire de-animation pipeline.
        '''
        canvas = StrokeCanvas(self.tracker.inputPath["anchor"], self.tracker.outdir)
        canvas.master.update()
        while True:
            if not canvas.destroy:
                canvas.master.update()
            else:
                break

        self.tracker.run()
        self.get_temporal_weights()
        self.pre_process()
        v_prime_anchor = self.energy_function_lsq()
        self.draw_rectangles(v_prime_anchor)
        self.texture_map(v_prime_anchor)

        self.tracker.mode = "floating"
        self.tracker.run()
        self.get_temporal_weights()
        self.pre_process()
        v_prime_floating = self.energy_function_lsq()
        self.draw_rectangles(v_prime_floating)
        self.texture_map(v_prime_floating)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", required=True, default="./config.yaml", 
        help="Path to config file")
    
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    tracker = Tracker(config)
    lsq_solver = LeastSquaresSolver(tracker)
    lsq_solver.run()

if __name__ == '__main__':
    main()
