import numpy as np
import scipy as sc
import scipy.sparse.linalg
import cv2
from math import sqrt
from klt import Tracker
import yaml
import argparse

class LeastSquaresSolver:
    def __init__(self, feature_table, vertices, weights_and_verts, quads):
        self.ka, self.vertices, self.weights_and_verts, self.non_solved_verts, self.processed_quads = self.pre_process(feature_table, vertices, weights_and_verts, quads)
        num_t = feature_table.shape[1]
        num_s = feature_table.shape[0]
        ka = self.ka
        vertices = self.vertices 
        weights_and_verts = self.weights_and_verts
        non_solved_verts = self.non_solved_verts
        processed_quads = self.processed_quads
        self.energy_function_lsq(num_s, num_t, ka, vertices, weights_and_verts, non_solved_verts, processed_quads)

    def pre_process(self, feature_table, vertices, weights_and_verts, quads):
        num_s = feature_table.shape[0]
        num_t = feature_table.shape[1]
        K_a = feature_table.transpose(1, 0, 2)[0].reshape(-1, 1)
        K_a = np.vstack([K_a for i in range(num_t)])
        assert(K_a.shape == (num_s*num_t*2, 1))

        processed_quads = np.zeros((64*32, 4))
        vertices = vertices.reshape(-1, 2)
        for i, quad in enumerate(quads):
            quad_idxs = []
            for vert in quad:
                quad_idxs.append(np.where((vertices == vert).all(axis=1))[0][0])
            processed_quads[i, :] = quad_idxs

        assert(processed_quads.shape == (64*32, 4))
        processed_quads = np.vstack([processed_quads for i in range(num_t)])
        assert(processed_quads.shape == (64*32*num_t, 4))
        for i in range(0, processed_quads.shape[0], 64*32):
            processed_quads[i:(64*32)*(int(i/2048) + 1), :] += len(vertices)*(int(i/2048))

        processed_quads = np.repeat(processed_quads, 2, axis=0)
        assert(processed_quads.shape == (64*32*2*num_t, 4))
        processed_quads *= 2
        processed_quads[np.arange(1, processed_quads.shape[0] + 1, 2)] += 1
        processed_quads = processed_quads.astype(np.int64)


        vertices = vertices.reshape(-1, 1)
        assert(vertices.shape == (65*33*2, 1))
        vertices = np.vstack([vertices for i in range(num_t)])
        assert(vertices.shape == (65*33*2*num_t, 1))
        weights_and_vert_idxs = weights_and_verts.reshape(-1, 2, 4)
        assert(weights_and_vert_idxs.shape == (num_s*num_t, 2, 4))
        weights_and_vert_idxs = np.repeat(weights_and_vert_idxs, 2, axis=0)
        assert(weights_and_vert_idxs.shape == (num_s*num_t*2, 2, 4))
        weights_and_vert_idxs[:, 1, :] *= 2
        x_coords = np.arange(1, weights_and_vert_idxs.shape[0] + 1, 2).astype(np.int64)
        weights_and_vert_idxs[x_coords, 1, :] += 1
        non_solved_verts = []
        vers = np.arange(0, vertices.shape[0], 1)
        reshaped_idxs = np.array(sorted(list(set(list(np.squeeze(weights_and_vert_idxs[:, 1, :].reshape(-1, 1)))))))
        for v in vers:
            if v not in reshaped_idxs:
                non_solved_verts.append(v)
        
        # print("Non Solved Verts Length: ", len(non_solved_verts))
        # print("Solved Verts Length: ", reshaped_idxs.shape[0])
        # print("Total Length: ", len(non_solved_verts) + reshaped_idxs.shape[0])
        # print("Lenght of all vertices", vertices.shape[0])
        # print("Ea", 65*33*2*num_t)
    
        return K_a, vertices, weights_and_vert_idxs, non_solved_verts, processed_quads

    def energy_function_lsq(self, num_s, num_t, ka, vertices, weights_and_verts, non_solved_verts, quads, l=None):
        #2*num_s*num_t -> E_a: each s has 4 x coord weights and 4 y coord weights (each set of weights has a row)..total of t frames
        #2*num_s*8*num_t -> E_s: each s has 4 points associated with it (quad), each point has 2 coordinates (x coord, y coord), each coord is part of 2 equations
        ea_rows = ka.shape[0] + len(non_solved_verts)
        es_rows = 64*32*8*2*num_t

        ka_final = np.zeros((ea_rows+es_rows, 1))
        ka_final[:ka.shape[0]] = ka

        #new matrix that will be A in Ax=b
        A = np.zeros((ea_rows+es_rows, len(vertices)))

        #Create K_a equations
        for i in range(0, weights_and_verts.shape[0]):
            weights = weights_and_verts[i, 0]
            vert_idxs = weights_and_verts[i, 1].astype(np.int64)
            A[i, vert_idxs] = weights
        
        #Create remaining equations
        for idx, vert in enumerate(non_solved_verts):
            offset = weights_and_verts.shape[0]
            A[idx + offset, vert] = 1
            ka_final[idx + offset] = vert

        ea_offset = ea_rows
        for i in range(0, es_rows, 16):
            quad_index = int(i/16)
            y1 = quads[quad_index, 0]
            x1 = quads[quad_index + 1, 0]

            y2 = quads[quad_index, 1]
            x2 = quads[quad_index + 1, 1]
            
            y3 = quads[quad_index, 2]
            x3 = quads[quad_index + 1, 2]

            y4 = quads[quad_index, 3]
            x4 = quads[quad_index + 1, 3]
            
            v1 = np.array([y1, x1])
            v2 = np.array([y2, x2])
            v3 = np.array([y3, x3])
            v4 = np.array([y4, x4])
            
            #Solve for index 0 using index 1 and 2 where index 1 is opposite hypotenuse
            combos_x = np.array([[x1, x2, x4], [x4, x2, x1], [x1, x3, x4], [x4, x3, x1], [x2, x1, x3], [x3, x1, x2], [x2, x4, x3], [x3, x4, x2]]).astype(np.int64)
            combos_y = np.array([[y1, y2, y4], [y4, y2, y1], [y1, y3, y4], [y4, y3, y1], [y2, y1, y3], [y3, y1, y2], [y2, y4, y3], [y3, y4, y2]]).astype(np.int64)
            combos = [[v1, v2, v4], [v4, v2, v1], [v1, v3, v4], [v4, v3, v1], [v2, v1, v3], [v3, v1, v2], [v2, v4, v3], [v3, v4, v2]]
            # top left: v1
            # top right: v2
            # bottom left: v3
            # bottom right: v4
            # [x1 y1] = [x2 y2] + [y4 - y2, x2 - x4]
            #x: index0x - index1x - index2y + index1y  
            #y: index0y - index1y - index1x + index2x
            for j in range(0, len(combos)*2, 2):
                x_idxs = np.array([combos_x[j//2, 0], combos_x[j//2, 1], combos_y[j//2, 2], combos_y[j//2, 1]])
                y_idxs = np.array([combos_y[j//2, 0], combos_y[j//2, 1], combos_x[j//2, 1], combos_x[j//2, 2]])

                x_coeffs = [1, -1, -1, 1]
                y_coeffs = [1, -1, -1, 1]

                A[i + j + ea_offset, y_idxs] = np.array(y_coeffs)
                A[i + j + 1 + ea_offset, x_idxs] = np.array(x_coeffs)

        print("Solving")
        new_w = scipy.sparse.lil_matrix(A, dtype='double')

        v_prime = scipy.sparse.linalg.lsqr(new_w.tocsr(), ka_final); # solve w/ csr
        v_prime = v_prime[0]


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

    feature_table, vertices, quads, quad_dict, quad_indices, weights_and_verts = tracker.run()
    print("called lsq")
    lsq_solver = LeastSquaresSolver(feature_table, vertices, weights_and_verts, quads)

if __name__ == '__main__':
    main()
