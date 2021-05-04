import numpy as np
import cv2
import scipy
from scipy import linalg

def texture_mapping(orig_quads, warped_quads, input_image):
    vid = cv2.Video(input_video)
    frame_counter = 0
    while(True):
        ret, frame = vid.read()
        if frame is None:
            break

        warped_frame = np.zeros_like(frame)
        
        frame_orig_qs = orig_quads[frame_counter*2048 : (frame_counter + 1)*2048]
        frame_warped_qs = warped_quads[frame_counter*2048: (frame_counter + 1)*2048]


        for i in range(len(frame_orig_qs)):
            o_quad = frame_orig_qs[i]
            w_quad = frame_warped_qs[i]

            u1 = o_quad[0][0]
            v1 = o_quad[0][1]

            u2 = o_quad[1][0]
            v2 = o_quad[1][1]

            u3 = o_quad[2][0]
            v3 = o_quad[2][1]

            u4 = o_quad[3][0]
            v4 = o_quad[3][1]

            u1_ = w_quad[0][0]
            v1_ = w_quad[0][1]

            u2_ = w_quad[1][0]
            v2_ = w_quad[1][1]

            u3_ = w_quad[2][0]
            v3_ = w_quad[2][1]

            u4_ = w_quad[3][0]
            v4_ = w_quad[3][1]

            A = [[-u1, -v1, -1, 0, 0, 0, u1*u1_, v1*u1_, u1_],
                 [0, 0, 0, -u1, -v1, -1, u1*v1_, v1*v1_, v1_],
                 [-u2, -v2, -1, 0, 0, 0, u2*u2_, v2*u2_, u2_],
                 [0, 0, 0, -u2, -v2, -1, u2*v2_, v2*v2_, v2_],
                 [-u3, -v3, -1, 0, 0, 0, u3*u3_, v3*u3_, u3_],
                 [0, 0, 0, -u3, -v3, -1, u3*v3_, v3*v3_, v3_],
                 [-u4, -v4, -1, 0, 0, 0, u4*u4_, v4*u4_, u4_],
                 [0, 0, 0, -u4, -v4, -1, u4*v4_, v4*v4_, v4_]]
            
            A = np.array(A)
            u,s,v_t = scipy.linalg.svd(A)
            T = v_t[-1]
            T = T.reshape((3,3))
            T_inv = np.linalg.inv(T)

        for y in range(v4_):
            for x in range(u4_):
                orig_point = np.dot(T_inv, np.array([y, x, 1]))
                #do I round orig_point?
                warped_intensity = frame[orig_point[0], orig_point[1], :] 
                warped_frame[y, x, :] = warped_intensity

        frame_counter += 1