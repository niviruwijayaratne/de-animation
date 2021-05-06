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

class LeastSquaresSolver:
    def __init__(self, tracker):
        self.tracker = tracker
        self.ka = None
        self.vertices = None
        self.weights_and_verts = None
        self.non_solved_verts = None 
        self.processed_quads = None
        self.temporal_weight_mat = None
        
        self.get_temporal_weights()
        self.pre_process()
        v_prime = self.energy_function_lsq()
        self.draw_rectangles(v_prime)
        self.texture_map(v_prime)

    def pre_process(self):
        num_s = self.tracker.feature_table.shape[0]
        num_t = self.tracker.feature_table.shape[1]
        self.ka = self.tracker.feature_table.transpose(1, 0, 2)[0].reshape(-1, 2)
        self.ka = self.tracker.feature_table.transpose(1, 0, 2)[0].reshape(-1, 1)
        self.ka = np.tile(self.ka, (num_t, 1))
        assert(self.ka.shape == (num_s*num_t*2, 1))

        self.processed_quads = np.zeros((64*32, 4))
        self.tracker.vertices = self.tracker.vertices.reshape(-1, 2)
        for i, quad in enumerate(self.tracker.quads):
            quad_idxs = []
            for vert in quad:
                quad_idxs.append(np.where((self.tracker.vertices == vert).all(axis=1))[0][0])
            self.processed_quads[i, :] = quad_idxs

        assert(self.processed_quads.shape == (64*32, 4))
        self.processed_quads = np.tile(self.processed_quads, (num_t, 1))
        assert(self.processed_quads.shape == (64*32*num_t, 4))
        for i in range(0, self.processed_quads.shape[0], 64*32):
            self.processed_quads[i:(64*32)*(int(i/2048) + 1), :] += len(self.tracker.vertices)*(int(i/2048))
        self.processed_quads = np.repeat(self.processed_quads, 2, axis=0)
        assert(self.processed_quads.shape == (64*32*2*num_t, 4))
        self.processed_quads *= 2
        self.processed_quads[np.arange(1, self.processed_quads.shape[0] + 1, 2)] += 1
        self.processed_quads = self.processed_quads.astype(np.int64)

        self.vertices = self.tracker.vertices.reshape(-1, 1)
        assert(self.vertices.shape == (65*33*2, 1))
        self.vertices = np.tile(self.vertices, (num_t, 1))
        assert(self.vertices.shape == (65*33*2*num_t, 1))
        self.weights_and_verts = self.tracker.weights_and_verts.reshape(-1, 2, 4)
        assert(self.weights_and_verts.shape == (num_s*num_t, 2, 4))
        self.weights_and_verts = np.repeat(self.weights_and_verts, 2, axis=0)
        assert(self.weights_and_verts.shape == (num_s*num_t*2, 2, 4))
        self.weights_and_verts[:, 1, :] *= 2
        x_coords = np.arange(1, self.weights_and_verts.shape[0] + 1, 2).astype(np.int64)
        self.weights_and_verts[x_coords, 1, :] += 1
        self.non_solved_verts = []
        vers = np.arange(0, self.vertices.shape[0], 1)
        reshaped_idxs = np.unique(self.weights_and_verts[:, 1, :].reshape(-1, 1), axis=0)
        for v in vers:
            if v not in reshaped_idxs:
                self.non_solved_verts.append(v)
        self.non_solved_verts = np.array(self.non_solved_verts).reshape(-1, 1)
        self.temporal_weight_mat = self.temporal_weight_mat.transpose(1, 0).reshape(-1, 1)
        self.temporal_weight_mat = np.repeat(self.temporal_weight_mat, 2, axis=0)

    def draw_rectangles(self, v_prime):
        print("drawing")
        f = cv2.imread('inputs/reference_frame.jpg')
        vid = cv2.VideoCapture(self.tracker.inputPath)
        count = 0
        h, w = self.tracker.reference_frame.shape[0], self.tracker.reference_frame.shape[1]
        writer = cv2.VideoWriter("results/bai_guitar_mesh.mp4" ,cv2.VideoWriter_fourcc(*'mp4v'), vid.get(cv2.CAP_PROP_FPS), (w, h))
        counter = 0
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

            cv2.line(f, (x_tl, y_tl), (x_tr, y_tr), (255, 0, 0), 1)
            cv2.line(f, (x_tr, y_tr), (x_br, y_br), (255, 0, 0), 1)
            cv2.line(f, (x_br, y_br), (x_bl, y_bl), (255, 0, 0), 1)
            cv2.line(f, (x_bl, y_bl), (x_tl, y_tl), (255, 0, 0), 1)

            y_tl_o = self.vertices[self.processed_quads[i][0]]
            x_tl_o = self.vertices[self.processed_quads[i + 1][0]]
            
            y_tr_o = self.vertices[self.processed_quads[i][1]]
            x_tr_o = self.vertices[self.processed_quads[i + 1][1]]

            y_bl_o = self.vertices[self.processed_quads[i][2]]
            x_bl_o = self.vertices[self.processed_quads[i + 1][2]]

            y_br_o = self.vertices[self.processed_quads[i][3]]
            x_br_o = self.vertices[self.processed_quads[i + 1][3]]

            if (i + 2) % (64*32*2) == 0:
                counter += 1
                writer.write(f)
                f = cv2.imread('inputs/reference_frame.jpg')
        writer.release()
    
    def get_temporal_weights(self):
        '''
        Input: num_features x num_frames x 2 matrix
        Return: num_features x num_frames matrix where each row corresponds to a feature and 
        each index gives the temporal weight of the feature at that frame index
        '''
        s, t = self.tracker.feature_table.shape[0], self.tracker.feature_table.shape[1]
        self.temporal_weight_mat = np.zeros((s, t))
        T = 15
        eps = 1e-2
        for s_i in range(s):
            time_frame = np.where(self.tracker.feature_table[s_i] != None)
            t_start = time_frame[0][0]
            t_end = time_frame[0][-1]
            
            for t_i in range(t):
                #check if track
                feature = self.tracker.feature_table[s_i, t_i]
                if (feature != None).all():
                    #apply piecewise function
                    if ( (t_i >= t_start) and (t_i < t_start + T) ):
                        self.temporal_weight_mat[s_i, t_i] = ((t_i - t_start) / T)
                    elif ( (t_i >= t_start + T) and (t_i <= t_end - T) ):
                        self.temporal_weight_mat[s_i, t_i] = 1
                    else:
                        self.temporal_weight_mat[s_i, t_i] = ((t_end - t_i) / T)

    def energy_function_lsq(self):
        num_s = self.tracker.feature_table.shape[0]
        num_t = self.tracker.feature_table.shape[1]
        ea_rows = self.ka.shape[0] + self.non_solved_verts.shape[0]     
        es_rows = 64*32*8*2*num_t

        ka_final = np.zeros((ea_rows+es_rows, 1))
        ka_final[:self.ka.shape[0]] = np.multiply(self.ka, np.sqrt(self.temporal_weight_mat))
        A = scipy.sparse.lil_matrix((ea_rows + es_rows, len(self.vertices)))
        
        #Create K_a equations
        print("Started E_a Equations")
        idxs = np.arange(0, self.weights_and_verts.shape[0], 1).reshape(-1, 1)
        vert_idxs = np.squeeze(self.weights_and_verts[idxs, 1].astype(np.int64))
        weights = np.squeeze(self.weights_and_verts[idxs, 0])
        A[idxs, vert_idxs] = np.multiply(weights, np.sqrt(self.temporal_weight_mat))
        
        print("Started NSV Equations")
        idxs = np.arange(self.weights_and_verts.shape[0], self.non_solved_verts.shape[0] + self.weights_and_verts.shape[0], 1).reshape(-1, 1)
        A[idxs, self.non_solved_verts] = 1
        ka_final[idxs] = self.vertices[self.non_solved_verts]
        
        print("Started E_s Equations")
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
        
        print("Solving")
        v_prime = scipy.sparse.linalg.lsqr(A.tocsr(), ka_final) # solve w/ csr
        v_prime = v_prime[0]
        
        z = 0
        for i in range(0, len(v_prime), 2):
            if v_prime[i] == 0:
                z += 1
            if v_prime[i + 1] == 0:
                z += 1
            if v_prime[i] > self.tracker.reference_frame.shape[0]:
                z += 1
            if v_prime[i + 1] > self.tracker.reference_frame.shape[1]:
                z += 1
        
        print(str(100 - (z/len(v_prime)*100)) + "% Valid Points")
        return v_prime

    def texture_map(self, v_prime):
        print("texture mapping")
        vid = cv2.VideoCapture(self.tracker.inputPath)
        ret, frame = vid.read()    
        writer = cv2.VideoWriter("results/bai_guitar.mp4" ,cv2.VideoWriter_fourcc(*'mp4v'), vid.get(cv2.CAP_PROP_FPS), (frame.shape[1], frame.shape[0]))
        warped_frame = np.zeros_like(frame)
        max_x = 0
        max_y = 0
        max_px = 0
        max_py = 0
        counter = 0
        ns = False
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
            for y in range(int(np.rint(y_tl)), int(np.rint(y_bl))):
                for x in range(int(np.rint(x_tl)), int(np.rint(x_tr))):
                    transformed_point = np.dot(transform, np.array([y, x, 1]))
                    transformed_point /= transformed_point[-1]
                    transformed_point = np.rint(transformed_point).astype(np.int64)
                    transformed_point[np.where(transformed_point < 0)] = 0
                    transformed_point[0] = min(transformed_point[0], frame.shape[0] - 1)
                    transformed_point[1] = min(transformed_point[1], frame.shape[1] - 1)
                    # print(str([y, x]) + " -> " + str(transformed_point))
                    # cv2.imshow('f', frame)
                    # cv2.waitKey(0)
                    intensity = frame[transformed_point[0], transformed_point[1], :] 
                    warped_frame[y, x, :] = intensity
                    # if max(max_x, transformed_point[1]) > max_x:
                    #     max_x = max(max_x, transformed_point[1])
                    #     max_px = x
                    # if max(max_y, transformed_point[0]) > max_y:
                    #     max_y = max(max_y, transformed_point[0])
                    #     max_py = y
            # writer.write(frame)
            if (i + 2) % (64*32*2) == 0:
                print("Frame: ", counter)
                writer.write(warped_frame)
                counter += 1
                # writer.release()
                # sys.exit()
                ret, frame = vid.read()
        writer.release()
        vid.release()

    def get_transform(self, src, dst):
        A = np.zeros((src.shape[0]*2, 9))
        for i in range(0, A.shape[0], 2):
            u, v = np.squeeze(src[int(i/2)])
            u_, v_ = np.squeeze(dst[int(i/2)])

            A[i] = [-u, -v, -1, 0, 0, 0, u*u_, v*u_, u_]
            A[i + 1] = [0, 0, 0, -u, -v, -1, u*v_, v*v_, v_]

        U,S,V_t = scipy.linalg.svd(A)
        T = V_t[-1]
        T = T.reshape((3,3))
        T_inv = np.linalg.inv(T)
        return T_inv
        
            
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", required=True, default="./config.yaml", 
        help="Path to config file")
    
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    tracker = Tracker(config)
    print("called run")
    tracker.run()
    print("called lsq")
    lsq_solver = LeastSquaresSolver(tracker)

if __name__ == '__main__':
    main()
