import numpy as np
np.set_printoptions(threshold=np.inf)

path = "./bf_out.npy"
data = np.load(path)
print(data)