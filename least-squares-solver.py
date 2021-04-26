#l - weighting function
#k(s,t) - each track s in frame t
#V(s,t) - 4x1 vector of the quad that enclose track s in frame t
#w(s,t) - 4x1 vector of corresponding weights for each of the vertices in V(s,t) such that w(s,t)*V(s,t) = K(s, ta)
# 
import numpy as np
import scipy as sc
import scipy.sparse.linalg
import cv2

def energy_function_lsq(s, v, w, ka, l):

    vertex_map={}

    #new matrix with mesh size 64x32
    #65*33*2 because must account for both x and y coordinates for each vertex
    new = np.zeros((2*s*t, 65*33*2))

    #map each vertex in 64x32 mesh to a vertex number
    count = 0
    for i in range(65):
        for j in range(33):
            vertex_map[count] = [j,i]
            count += 1
    print("vertex_map: ")
    print(vertex_map)
    print()
    print("points and corresponding vertex number (using map):")

    #using weights and their coordinates from w, construct a new w that has weights in the correct places 
    #with the correct mappings
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
        t_start_idx += 4096
    print()
    print("final matrix:")
    print(new)

    new_w = scipy.sparse.lil_matrix(new, dtype='double')

    v_prime = scipy.sparse.linalg.lsqr(new_w.tocsr(), ka); # solve w/ csr
    v_prime = v_prime[0]

    print(v_prime)





