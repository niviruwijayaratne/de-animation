import numpy as np
import scipy as sc
import scipy.sparse.linalg
import cv2
from math import sqrt

#INPUTS
#quad_matrix: quads enclosing s where points are in order top left, top right, bottom right, bottom left
#num_t: number of total frames
#num_s: number of total features per frame (anchor tracks)
#v_map: potential vertext map??
#v: input mesh coordinates for quad vertices enclosing anchor features
#w: matrix with weights and x/y coordinate tuples for quads enclosing anchor features
#ka: vector with 2*num_s*num_t rows, each entry corresponding to the x/y coordinate of an anchor feature
#l: weighting function 

def energy_function_lsq(quad_matrix, num_t, num_s, v, w, ka, l):
    
    #2*num_s*num_t -> E_a: each s has 4 x coord weights and 4 y coord weights (each set of weights has a row)..total of t frames
    #2*num_s*8*num_t -> E_s: each s has 4 points associated with it (quad), each point has 2 coordinates (x coord, y coord), each coord is part of 2 equations
    ea_rows = 2*num_s*num_t
    es_rows = 4*2*2*num_s*num_t

    #Generate new K_a matrix with dimensions ea_rows+es_rows x 1. First E_a rows in K_a should match input. The rest should be 0
    ka_final = np.zeros((ea_rows+es_rows,1))

    for i in range(ea_rows):
        ka_final[i] = ka[i]

    #new matrix that will be A in Ax=b
    new = np.zeros((ea_rows+es_rows, len(v)))
    
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
    t_start_idx = 0
    for t in range(num_t):
        for i in range(0,w.shape[0], 2):
            for j in range(w.shape[1]):
                
                if i % 2 == 0:
                    i = t_start_idx+i

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

        t_start_idx += num_s*2 #rows for start of next frame

    # r90 = np.array([[0,1],[-1,0]])

    #helps gets index of row in new matrix where values should be filled in
    count = 0
    ea_offset = 2*num_s*num_t

    for t in range(num_t):
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





