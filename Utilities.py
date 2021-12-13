import numpy as np
import scipy as sp
from scipy import stats
import matplotlib.pyplot as plt
from scipy.fftpack import fft, fftshift, ifft, ifftshift
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
import pylab
from scipy.stats import wasserstein_distance
from scipy.sparse import csgraph
from scipy.sparse.linalg import eigsh
from scipy.spatial.distance import pdist, squareform
import numpy as np
from scipy.stats.kde import gaussian_kde
from sklearn.metrics.pairwise import euclidean_distances, manhattan_distances
import decimal
import warnings
import numpy as np
from math import radians, cos, sin, asin, sqrt

def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 3956  # Radius of earth in kilometers. Use 3956 for miles
    return c * r

def generate_olympic_data(grid):
    indicator_array = []
    for i in range(len(grid)):
        if grid[i] % 4 == 0:
            indicator_array.append(1)
        else:
            indicator_array.append(0)
    return np.asarray(indicator_array)

def indicator_function(grid):
    indicator_array = []
    for i in range(len(grid)):
        remainder = grid[i] % 4
        indicator_array.append(remainder)
    return np.asarray(indicator_array)

def dendrogram_plot_labels(matrix, distance_measure, data_generation, labels):

    # Compute and plot dendrogram.
    plt.rcParams.update({'font.size': 12})
    fig = pylab.figure(figsize=(15,10))
    axdendro = fig.add_axes([0.11,0.1,0.2,0.8])
    Y = sch.linkage(matrix, method='centroid')
    Z = sch.dendrogram(Y, orientation='right', labels=labels, leaf_rotation=360, leaf_font_size=12, show_leaf_counts=False)
    # axdendro.set_xticks([])
    # axdendro.set_yticks([])

    # Plot distance matrix.
    axmatrix = fig.add_axes([0.3,0.1,0.6,0.8])
    index = Z['leaves']
    D = matrix[index,:]
    D = D[:,index]
    im = axmatrix.matshow(D, aspect='auto', origin='lower')
    axmatrix.set_xticks([])
    axmatrix.set_yticks([])

    # Plot colorbar.
    axcolor = fig.add_axes([0.91,0.1,0.02,0.8])
    pylab.colorbar(im, cax=axcolor)
    # plt.title(data_generation+distance_measure+"Dendrogram")
    plt.savefig(data_generation+distance_measure+"Dendrogram")
    plt.show()

    # Display and save figure.
    fig.show()

def gini(x):
    # (Warning: This is a concise implementation, but it is O(n**2)
    # in time and memory, where n = len(x).  *Don't* pass in huge
    # samples!)

    # Mean absolute difference
    mad = np.abs(np.subtract.outer(x, x)).mean()
    # Relative mean absolute difference
    rmad = mad/np.mean(x)
    # Gini coefficient
    g = np.nan_to_num(0.5 * rmad)
    return g

from typing import List
from itertools import combinations
import numpy as np

def gini_coefficient(x: List[float]) -> float:
    x = np.array(x, dtype=np.float32)
    n = len(x)
    diffs = sum(abs(i - j) for i, j in combinations(x, r=2))
    return diffs / (2 * n**2 * x.mean())