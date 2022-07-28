import numpy as np
from ray.rllib.evaluation.postprocessing import discount_cumsum

x = np.array([0.0, 1.0, 2.0, 3.0])
gamma = 0.9
ret = discount_cumsum(x, gamma)
print(ret)