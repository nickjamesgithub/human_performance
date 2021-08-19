import numpy as np

def generate_olympic_data(grid):
    indicator_array = []
    for i in range(len(grid)):
        if grid[i] % 4 == 0:
            indicator_array.append(1)
        else:
            indicator_array.append(0)
    return np.asarray(indicator_array)