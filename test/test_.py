import ray
from ray.rllib.utils.framework import try_import_tf, try_import_torch
tf1, tf, tfv = try_import_tf()
torch, _ = try_import_torch()

print(tf)
print(torch.randn(3,4))