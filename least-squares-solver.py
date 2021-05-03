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
        # print("KA: ", sys.getsizeof(self.ka))
        # print("Vertices: ", sys.getsizeof(self.vertices))
        # print("WV: ", sys.getsizeof(self.weights_and_verts))
        # print("NSV: ", sys.getsizeof(self.non_solved_verts))
        # print("QS: ", sys.getsizeof(self.processed_quads))
        # print("Tracker: ", sys.getsizeof(self.tracker))
        # sys.exit()
        

        v_prime = self.energy_function_lsq()
        # x = v_prime[np.array(self.non_solved_verts)] - self.vertices[np.array(self.non_solved_verts)]
        # for x in self.non_solved_verts:
        #     print(v_prime[x], self.vertices[x])

        # sys.exit()
        self.draw_rectangles(v_prime)

    def pre_process(self):
        num_s = self.tracker.feature_table.shape[0]
        num_t = self.tracker.feature_table.shape[1]
        self.ka = self.tracker.feature_table.transpose(1, 0, 2)[0].reshape(-1, 2)
        self.ka = self.tracker.feature_table.transpose(1, 0, 2)[0].reshape(-1, 1)
        self.ka = np.vstack([self.ka for i in range(num_t)])
        assert(self.ka.shape == (num_s*num_t*2, 1))

        self.processed_quads = np.zeros((64*32, 4))
        self.tracker.vertices = self.tracker.vertices.reshape(-1, 2)
        for i, quad in enumerate(self.tracker.quads):
            quad_idxs = []
            for vert in quad:
                quad_idxs.append(np.where((self.tracker.vertices == vert).all(axis=1))[0][0])
            self.processed_quads[i, :] = quad_idxs

        assert(self.processed_quads.shape == (64*32, 4))
        self.processed_quads = np.vstack([self.processed_quads for i in range(num_t)])
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
        self.vertices = np.vstack([self.vertices for i in range(num_t)])
        assert(self.vertices.shape == (65*33*2*num_t, 1))
        self.weights_and_verts = self.tracker.weights_and_verts.reshape(-1, 2, 4)
        assert(self.weights_and_verts.shape == (num_s*num_t, 2, 4))
        self.weights_and_verts = np.repeat(self.weights_and_verts, 2, axis=0)
        assert(self.weights_and_verts.shape == (num_s*num_t*2, 2, 4))
        self.weights_and_verts[:, 1, :] *= 2
        x_coords = np.arange(1, self.weights_and_verts.shape[0] + 1, 2).astype(np.int64)
        self.weights_and_verts[x_coords, 1, :] += 1
        self.non_solved_verts = []
        vers = np.arange(0, self.tracker.vertices.shape[0], 1)
        reshaped_idxs = np.array(sorted(list(set(list(np.squeeze(self.weights_and_verts[:, 1, :].reshape(-1, 1)))))))
        for v in vers:
            if v not in reshaped_idxs:
                self.non_solved_verts.append(v)

        self.temporal_weight_mat = self.temporal_weight_mat.transpose(1, 0).reshape(-1, 1)
        self.temporal_weight_mat = np.repeat(self.temporal_weight_mat, 2, axis=0)

    def draw_rectangles(self, v_prime):
        print("drawing")
        f = self.tracker.reference_frame.copy()
        count = 0
        h, w = self.tracker.reference_frame.shape[0], self.tracker.reference_frame.shape[1]
        writer = cv2.VideoWriter("results/warped_quads.mp4" ,cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (w, h))

        for i in range(0, self.processed_quads.shape[0], 2):
            y_tl = v_prime[self.processed_quads[i][0]]
            x_tl = v_prime[self.processed_quads[i + 1][0]]

            y_br = v_prime[self.processed_quads[i][3]]
            x_br = v_prime[self.processed_quads[i + 1][3]]
            
            y_tl = int(np.rint(y_tl))
            x_tl = int(np.rint(y_tl))
            y_br = int(np.rint(y_br))
            x_br = int(np.rint(x_br))

            cv2.rectangle(f, (x_tl, y_tl), (x_br, y_br), (255, 0, 0), 1)

            y_tl_o = self.vertices[self.processed_quads[i][0]]
            x_tl_o = self.vertices[self.processed_quads[i + 1][0]]

            y_br_o = self.vertices[self.processed_quads[i][3]]
            x_br_o = self.vertices[self.processed_quads[i + 1][3]]

            if (i + 2) % (64*32*2) == 0:
                writer.write(f)
                # cv2.imshow('f', f)
                # cv2.waitKey(0)
                f = self.tracker.reference_frame.copy()
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
        
        for s_i in range(s):
            time_frame = np.where(self.tracker.feature_table[s_i] != 0)
            t_start = time_frame[0][0]
            t_end = time_frame[0][-1]
            for t_i in range(t):
                #check if track
                feature = self.tracker.feature_table[s_i, t_i]
                if (feature != None).all():
                    #apply piecewise function
                    if ( (t_i >= t_start) and (t_i < t_start + T) ):
                        self.temporal_weight_mat[s_i, t_i] = (t_i - t_start) / T
                    elif ( (t_i >= t_start + T) and (t_i <= t_end - T) ):
                        self.temporal_weight_mat[s_i, t_i] = 1
                    else:
                        self.temporal_weight_mat[s_i, t_i] = (t_end - t_i) / T

    def energy_function_lsq(self):
        num_s = self.tracker.feature_table.shape[0]
        num_t = self.tracker.feature_table.shape[1]
        ea_rows = self.ka.shape[0] + len(self.non_solved_verts)        
        es_rows = 64*32*8*2*num_t
        ka_final = np.zeros((ea_rows+es_rows, 1))
        ka_final[:self.ka.shape[0]] = np.multiply(self.ka, np.sqrt(self.temporal_weight_mat))
        A = scipy.sparse.lil_matrix((ea_rows + es_rows, len(self.vertices)))
        
        #Create K_a equations
        for i in range(0, self.weights_and_verts.shape[0]):
            weights = self.weights_and_verts[i, 0]
            vert_idxs = self.weights_and_verts[i, 1].astype(np.int64)
            A[i, vert_idxs] = weights*np.sqrt(self.temporal_weight_mat[i])
        
        #Create remaining equations
        for idx, vert in enumerate(self.non_solved_verts):
            offset = self.weights_and_verts.shape[0]
            A[idx + offset, vert] = 1
            ka_final[idx + offset] = self.vertices[vert]
        
        #Create K_s equations
        ea_offset = ea_rows
        for i in range(0, es_rows, 16):
            quad_index = int(i/8)
            # print(i, quad_index)
            y1 = self.processed_quads[quad_index, 0]
            x1 = self.processed_quads[quad_index + 1, 0]

            y2 = self.processed_quads[quad_index, 1]
            x2 = self.processed_quads[quad_index + 1, 1]
            
            y3 = self.processed_quads[quad_index, 2]
            x3 = self.processed_quads[quad_index + 1, 2]

            y4 = self.processed_quads[quad_index, 3]
            x4 = self.processed_quads[quad_index + 1, 3]
            
            v1 = np.array([y1, x1])
            v2 = np.array([y2, x2])
            v3 = np.array([y3, x3])
            v4 = np.array([y4, x4])
            
            #Solve for index 0 using index 1 and 2 where index 1 is opposite hypotenuse
            combos_x = np.array([[x1, x2, x4], [x4, x2, x1], [x1, x3, x4], [x4, x3, x1], [x2, x1, x3], [x3, x1, x2], [x2, x4, x3], [x3, x4, x2]]).astype(np.int64)
            combos_y = np.array([[y1, y2, y4], [y4, y2, y1], [y1, y3, y4], [y4, y3, y1], [y2, y1, y3], [y3, y1, y2], [y2, y4, y3], [y3, y4, y2]]).astype(np.int64)
            combos = [[v1, v2, v4], [v4, v2, v1], [v1, v3, v4], [v4, v3, v1], [v2, v1, v3], [v3, v1, v2], [v2, v4, v3], [v3, v4, v2]]
            
            # [x1, y1] = [x2, y2] + [[y3 - y2], [x2 - x3]]
            # x1 = x2 + y3 - y2
            # y1 = y2 + x2 - x3

            # x1 - x2 - y3 + y2 = 0
            # y1 - y2 - x2 + x3 = 0

            for j in range(0, len(combos)*2, 2):
                x_idxs = np.array([combos_x[j//2, 0], combos_x[j//2, 1], combos_y[j//2, 2], combos_y[j//2, 1]])
                y_idxs = np.array([combos_y[j//2, 0], combos_y[j//2, 1], combos_x[j//2, 1], combos_x[j//2, 2]])

                x_coeffs = [1, -1, -1, 1]
                y_coeffs = [1, -1, -1, 1]
                A[i + j + ea_offset, y_idxs] = np.array(y_coeffs)*np.sqrt(4)/np.sqrt(8)
                A[i + j + 1 + ea_offset, x_idxs] = np.array(x_coeffs)*np.sqrt(4)/np.sqrt(8)

        v_prime = scipy.sparse.linalg.lsqr(A.tocsr(), ka_final) # solve w/ csr
        v_prime = v_prime[0]

        return v_prime
        
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
