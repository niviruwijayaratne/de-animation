import numpy as np
import scipy as sc
import scipy.sparse.linalg
import cv2
from math import sqrt
from klt import Tracker
import yaml

#INPUTS
#quad_matrix: quads enclosing s where points are in order top left, top right, bottom right, bottom left s*t x 4
#num_t: number of total frames
#num_s: number of total features per frame (anchor tracks)
#v_map: potential vertext map??
#v: input mesh coordinates for quad vertices enclosing anchor features
#w: matrix with weights and x/y coordinate tuples for quads enclosing anchor features num_t*num_s*2 x 4
#ka: vector with 2*num_s*num_t rows, each entry corresponding to the x/y coordinate of an anchor feature
#l: weighting function 
#have to flatten at end
'''
Process for getting weight indices and ka
1. Reshape K_a ie. first row in feature track to have 1 column ie. [x1, y1, x2, y2] and so on
2. Repeat K_a, num_frames times, to get a column vector of shape 2*num_features*num_frames x 1
3. Reshape vertices to have 1 column ie. [v1x, v1y, v2x, v2y] of shape 2*65*33 x 1
4. Repeat vertices, num_frames times, to get a column vector of shape 2*65*33*num_frames x 1 (Need to store this)
5. Reshape weights_and_vertices to shape (-1, 1, 2) ie. weights_and_vertices should have shape num_feature*num_frames x 2 x 4
    where each row gives the weights associated with feature point of the same row (and row + 1) in K_a in 1., and it's corresponding quad vertex indices in 
    vertices created in 4. Will have to multiply each row second element (vertex indices) by 2 to get index value in vertices created in 4. Will have to add
    1 to the second element in every other row to give y coordinates. 
6. For every 2 rows, can recover (x, y) coordinates of quad made up from those points and find the combinations order is always top left, top right,
    bottom left, bottom right

    
'''
class LeastSquaresSolver:
    def __init__(self, feature_table, vertices, weights_and_verts):
        self.ka, self.vertices, self.weights_and_verts = self.pre_process(feature_table, vertices, weights_and_verts)
        

    def pre_process(self, feature_table, vertices, weights_and_verts):
        K_a = feature_table.transpose(1, 0, 2)[0].reshape(-1, 1)
        K_a = np.vstack([K_a for i in range(feature_table.shape[1])])
        assert(K_a.shape == (feature_table.shape[0]*feature_table.shape[1]*2, 1))
        
        vertices = vertices.reshape(-1, 1)
        assert(vertices.shape == (65*33*2, 1))
        vertices = np.vstack([vertices for i in range(feature_table.shape[1])])
        assert(vertices.shape == (65*33*2*feature_table.shape[1], 1))
        
        weights_and_vert_idxs = weights_and_verts.reshape(-1, 2, 4)
        assert(weights_and_vert_idxs.shape == (feature_table.shape[0]*feature_table.shape[1], 2, 4))
        weights_and_vert_idxs = np.repeat(weights_and_vert_idxs, 2, axis=0)
        assert(weights_and_vert_idxs.shape == (feature_table.shape[0]*feature_table.shape[1]*2, 2, 4))
        weights_and_vert_idxs[:, 1, :] *= 2
        y_coords = np.arange(1, weights_and_vert_idxs.shape[0] + 1, 2).astype(np.int64)
        weights_and_vert_idxs[y_coords, 1, :] += 1
        return K_a, vertices, weights_and_vert_idxs

    def energy_function_lsq(self):
        pass
    def energy_function_lsq(self, quad_matrix, num_t, num_s, v, w, ka, l=None):
    

        #2*num_s*num_t -> E_a: each s has 4 x coord weights and 4 y coord weights (each set of weights has a row)..total of t frames
        #2*num_s*8*num_t -> E_s: each s has 4 points associated with it (quad), each point has 2 coordinates (x coord, y coord), each coord is part of 2 equations
        ea_rows = 2*num_s*num_t
        es_rows = 4*2*2*num_s*num_t

        #Generate new K_a matrix with dimensions ea_rows+es_rows x 1. First E_a rows in K_a should match input. The rest should be 0
        ka_final = np.zeros((ea_rows+es_rows,1))
        ka_final[:self.ka.shape[0]] = self.ka

        #new matrix that will be A in Ax=b
        A = np.zeros((ea_rows+es_rows, len(v)))
        
        vertex_map={}
        cnt = 0
        for i in range(quad_matrix.shape[0]):
            for j in range(quad_matrix.shape[1]):
                vertex_map[cnt] = tuple(quad_matrix[i][j])
                cnt += 1
        
        #map each vertex in 64x32 mesh to a vertex number
        # vertex_map={}
        # count = 0
        # for i in range(65):
        #     for j in range(33):
        #         vertex_map[count] = [j,i]
        #         count += 1


        #using weights and their coordinates from w, construct a new w that has weights in the correct places 
        #with the correct mappings --> E_a
        for i in range(0,w.shape[0], 2):
            for j in range(w.shape[1]):
                if i % 2 == 0:
                    weight_x = w[i,j,0]
                    weight_y = w[i+1,j,0]
                    
                    x_coord = w[i,j,1]
                    y_coord = w[i+1,j,1]
                
                    #find vertex number the coordinate maps to...will use this number to find the index of 
                    #the row at which the x and y weights should be inserted
                    for key, value in vertex_map.items():
                        if value == [x_coord,y_coord]:
                            vertex_num = key
                    print([x_coord, y_coord], "->", vertex_num)
                    
                    if vertex_num == 0:
                        new[i,0] = weight_x
                        new[i+1,1] = weight_y
                        
                    else:  
                        new[i,vertex_num*2] = weight_x
                        new[i+1,vertex_num*2+1] = weight_y


        # r90 = np.array([[0,1],[-1,0]])

        #helps gets index of row in new matrix where values should be filled in
        count = 0
        ea_offset = 2*num_s*num_t

        for i in range(quad_matrix.shape[0]):
            x1 = quad_matrix[i][0][0]
            y1 = quad_matrix[i][0][1]

            x2 = quad_matrix[i][1][0]
            y2 = quad_matrix[i][1][1]

            x3 = quad_matrix[i][2][0]
            y3 = quad_matrix[i][2][1]

            x4 = quad_matrix[i][3][0]
            y4 = quad_matrix[i][3][1]
            
            
            v1 = np.array([x1, y1])
            v2 = np.array([x2,y2])
            v3 = np.array([x3,y3])
            v4 = np.array([x4,y4])
            
            #ex: solving for v1 using v4 and v3 where v4 is vertex opposite hypotenuse
            combos = [(v1,v4,v3), (v1,v2,v3), (v2,v1,v4), (v2,v3,v4), (v3,v4,v1), (v3,v2,v1), (v4,v3,v2), (v4,v1,v2)]
            # top left: v1
            # top right: v2
            # bottom right: v3

            for j in combos:
                x_coord = j[0][0]
                y_coord = j[0][1]
                
                x_coord2 = j[1][0]
                y_coord2 = j[1][1]
                
                x_coord3 = j[2][0]
                y_coord3 = j[2][1]
                
                for key, value in vertex_map.items():
                    if value == (x_coord,y_coord):
                        vertex_num1 = key
                    elif value == (x_coord2,y_coord2):
                        vertex_num2 = key
                    elif value == (x_coord3,y_coord3):
                        vertex_num3 = key  
                
                #FINDING EQUATIONS
                # V2 + R90(V3 − V2) ... V2 being vertex opposite hypotenuse, second vertex in all combos above
                #r90 transforms vector to -> [y -x]
                #[v1x v1y] = [v2x v2y] + [[v3y-v2y], [v2x-v3x]] 
                
                #X COORDINATE
                #v1x
                if vertex_num1 == 0:
                    new[ea_offset+count,0] = 1
                else:
                    new[ea_offset+count,vertex_num1*2] = 1
                    print(vertex_num1*2)
                #-v2x
                if vertex_num2 == 0:
                    new[ea_offset+count,0] = -1
                else:
                    new[ea_offset+count,vertex_num2*2] = -1

                #-v3y
                if vertex_num3 == 0:
                    new[ea_offset+count,1] = -1
                else:
                    new[ea_offset+count,vertex_num3*2+1] = -1

                #v2y
                if vertex_num2 == 0:
                    new[ea_offset+count,1] = 1
                else:
                    new[ea_offset+count,vertex_num2*2+1] = 1

                
                
                #Y COORDINATE
                #v1y
                if vertex_num1 == 0:
                    new[ea_offset+count+1,1] = 1
                else:
                    new[ea_offset+count+1,vertex_num1*2 + 1] = 1
                
                #-v2y
                if vertex_num2 == 0:
                    new[ea_offset+count+1,1] = -1
                else:
                    new[ea_offset+count+1,vertex_num2*2+1] = -1 
                
                #-v2x
                if vertex_num2 == 0:
                    new[ea_offset+count+1,0] = -1
                else:
                    new[ea_offset+count+1,vertex_num2*2] = -1
                
                #v3x
                if vertex_num3 == 0:
                    new[ea_offset+count+1,0] = 1
                else:
                    new[ea_offset+count+1,vertex_num3*2] = 1
                    
                count+=2

        new_w = scipy.sparse.lil_matrix(new, dtype='double')

        v_prime = scipy.sparse.linalg.lsqr(new_w.tocsr(), ka_final); # solve w/ csr
        v_prime = v_prime[0]

def main():
    with open('config.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    tracker = Tracker(config)
    feature_table = tracker.track_features()
    vertices, quads, quad_dict = tracker.get_mesh() #all x,y pairs, all quads, dictionary where key = vertex.tobytes(), value = indices of quads in quads it belongs to
    quad_indices = tracker.search_quads(feature_table, quad_dict) #indices of each quad to be solved for in quads
    weights_and_verts = tracker.get_weights(feature_table, quads, quad_indices, vertices) #list of weights and their corresponding vertices
    ka = feature_table.transpose(1, 0, 2)[0]
    quad_matrix = quads
    num_t = ka.shape[0]
    num_s = ka.shape[1]

    lsq_solver = LeastSquaresSolver(feature_table, vertices, weights_and_verts)
    # energy_function_lsq(quad_matrix, num_t, num_s, vertices, weights_and_verts, ka)  


main()
