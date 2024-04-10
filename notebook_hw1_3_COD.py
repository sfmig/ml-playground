# %%
# Imports

import numpy as np
import matplotlib.pyplot as plt

# %%
# Input data
dim = 1000
radius = 1
n_samples_in = 100

# %%%%
# Sample points from a cube centred at the origin and length=2
# Discard points if outside the sphere

x_selected_samples = np.zeros((n_samples_in, dim))

n_samples_from_cube = 0
idx_in_sphere = 0
while idx_in_sphere <= (n_samples_in-1):
    # Sample one point in the cube
    x_vec = np.random.default_rng().uniform(-1, 1, dim).reshape(1, dim)
    n_samples_from_cube += 1
    
    # Check distance to origin
    if np.sum(x_vec**2, axis=1) > radius:
        continue
    # Save if point is in sphere
    else:
        x_selected_samples[idx_in_sphere,:] = x_vec
        idx_in_sphere += 1

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%
print(f'Points sampled from (hyper)cube: {n_samples_from_cube}')
# %%
