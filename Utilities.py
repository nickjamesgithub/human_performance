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


def generate_olympic_data(grid):
    indicator_array = []
    for i in range(len(grid)):
        if grid[i] % 4 == 0:
            indicator_array.append(1)
        else:
            indicator_array.append(0)
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
