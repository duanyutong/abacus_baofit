import numpy as np
data = np.fromfile('sample_input_gal_mx3_float32_0-1100_mpch_rsd.dat', dtype=np.float32).reshape(-1, 4)  # data array is of shape (ND, 4) for mx3 input
ND = data.shape[0]  # number of galaxy data