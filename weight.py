import numpy as np
def compute_weight_matrix(input_matrix):
    row, col = input_matrix.shape
    output_matrix = np.zeros((input_matrix.shape))
    T = 15
    
    for s in range(row):
        #get t_a and t_b
        time_frame = np.where(input_matrix[s] == 1)
        t_a = time_frame[0][0]
        t_b = time_frame[0][-1]
            
        for t in range(col):
            #check if track
            x = input_matrix[s][t]
            if (x == 1):
                #apply piecewise function
                if ( (t >= t_a) and (t < t_a + T) ):
                    output_matrix[s,t] = (t - t_a) / T
                elif ( (t >= t_a + T) and (t <= t_b - T) ):
                    output_matrix[s,t] = 1
                else:
                    output_matrix[s,t] = (t_b - t) / T

    return output_matrix

