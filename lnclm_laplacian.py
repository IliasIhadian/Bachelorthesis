import numba
import scipy.sparse.linalg
import scipy.sparse
import numpy as np
import scipy.sparse
from numba import njit
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from util.kdtree import *


def normalize_rows(A, threshold=0.0):
    """Normalize the rows of a matrix

    Rows with sum below threshold are left as-is.

    Parameters
    ----------
    A: scipy.sparse.spmatrix
        Matrix to normalize
    threshold: float
        Threshold to avoid division by zero

    Returns
    -------
    A: scipy.sparse.spmatrix
        Matrix with normalized rows

    Example
    -------
    >>> from pymatting import *
    >>> import numpy as np
    >>> A = np.arange(4).reshape(2,2)
    >>> normalize_rows(A)
    array([[0. , 1. ],
           [0.4, 0.6]])
    """
    row_sums = A.dot(np.ones(A.shape[1], A.dtype))

    # Prevent division by zero.
    row_sums[row_sums < threshold] = 1.0

    row_normalization_factors = 1.0 / row_sums

    D = scipy.sparse.diags(row_normalization_factors)

    A = D.dot(A)

    return A


@njit("void(f8[:, :, :], f8, f8[:], i8[:], i8[:], i8, i8[:,:], b1[:])", cache=True, nogil=True, parallel=True)
def calc(image, epsilon, v, i, j,  neighbour_count, neighbours, is_known):
    _, w, _ = image.shape

    ykoos = neighbours // w
    xkoos = neighbours % w

    nnimage = np.zeros((len(ykoos),neighbour_count, 3))
    for pixel in numba.prange(len(ykoos)):

        if np.all(is_known[neighbours[pixel]]):
            continue

        # hier packen wir die pixel des bildes von den nearestneighbours in ein array
        for c_tmp in range(neighbour_count):
            # hier holen wir die image pixel raus
            nnimage[pixel][c_tmp] = image[ykoos[pixel, c_tmp], xkoos[pixel, c_tmp]]
        
        # Hier berechnen wir mu
        mu = np.zeros(3)
        for dc in range(3):
            # Hier berechnen wir den durchnitt von jeder Farben-Achse der NearestNeighbours des pixels (x,y)
            mu[dc] = np.mean(nnimage[pixel][:, dc])

        # Hier berechnen wir I - mu innerhalb des Fensters
        c = nnimage[pixel] - mu
        
        # Hier berechnen wir die 3x3 Kovarianzmatrix für die verschiedenen Farb-achsen
        cov = np.zeros((3, 3))
        for p in range(3):
            for q in range(3):
                cov[p, q] = np.mean(c[:, p] * c[:, q])
        
        # Hier addieren wir die cov mit einer 3x3 Diagonalmatrix mit epsilon/window_area (Da cov 3x3)
        cov_tmp = cov + epsilon / neighbour_count * np.eye(3)
        
        #Hier berechnen wir die Inverse der cov_tmp
        #hier schneller machen indem du auschreibst guck pymtting 
        inv = np.linalg.inv(cov_tmp)
        
        # Hier gehen wir mit 2 Pixeln (dyi,dxi) (dyj,dxj) durch das Fenster
        for di in range(neighbour_count):
            for dj in range(neighbour_count):
                index = pixel*(neighbour_count**2) + di*neighbour_count + dj

                
                i[index] = ykoos[pixel, di]*w + xkoos[pixel, di]
                j[index] = ykoos[pixel, dj]*w + xkoos[pixel, dj]


                # Hier berechnen wir das skalarprodukt von (I_i - mu_k)(inv_cov)(I_j - mu_k)
                tmp = c[di].dot(inv).dot(c[dj])
                
                # Hier beenden wir die Formel für die LM
                v[index] = (1.0 if (i[index] == j[index]) else 0.0) - (1 + tmp) / len(ykoos[pixel])

#diese methode entstammt von https://github.com/pymatting/pymatting/blob/master/pymatting/util/util.py
def _knn_laplacian(
    image,
    n_neighbors=[20, 10],
    distance_weights=[2.0, 0.1],
    kernel="binary",
):
    """
    This function calculates the KNN matting Laplacian matrix similar to
    :cite:`chen2013knn`.
    We use a kernel of 1 instead of a soft kernel by default since the former is
    faster to compute and both produce almost identical results in all our
    experiments, which is to be expected as the soft kernel is very close to 1
    in most cases.

    Parameters
    ----------
    image: numpy.ndarray
        Image with shape :math:`h\\times w \\times 3`
    n_neighbors: list of ints
        Number of neighbors to consider. If :code:`len(n_neighbors)>1` multiple
        nearest neighbor calculations are done and merged, defaults to
        `[20, 10]`, i.e. first 20 neighbors are considered and in the second run
        :math:`10` neighbors. The pixel distances are then weighted by the
        :code:`distance_weights`.
    distance_weights: list of floats
        Weight of distance in feature vector, defaults to `[2.0, 0.1]`.
    kernel: str
        Must be either "binary" or "soft". Default is "binary".

    Returns
    -------
    L: scipy.sparse.spmatrix
        Matting Laplacian matrix
    """
    h, w = image.shape[:2]
    r, g, b = image.reshape(-1, 3).T
    n = w * h

    if kernel not in ["binary", "soft"]:
        raise ValueError("kernel must be binary/soft, but not " + kernel + ".")

    x = np.tile(np.linspace(0, 1, w), h)
    y = np.repeat(np.linspace(0, 1, h), w)

    # Store weight matrix indices and values in sparse coordinate form.
    i, j, coo_data = [], [], []

    for k, distance_weight in zip(n_neighbors, distance_weights):
        # Features consist of RGB color values and weighted spatial coordinates.
        f = np.stack(
            [r, g, b, distance_weight * x, distance_weight * y],
            axis=1,
            out=np.zeros((n, 5), dtype=np.float32),
        )

        # Find indices of nearest neighbors in feature space.
        _, neighbor_indices = knn(f, f, k=k)

        # [0 0 0 0 0 (k times) 1 1 1 1 1 2 2 2 2 2 ...]
        i.append(np.repeat(np.arange(n), k))
        j.append(neighbor_indices.ravel())

        W_ij = np.ones(k * n)

        if kernel == "soft":
            W_ij -= np.abs(f[i[-1]] - f[j[-1]]).sum(axis=1) / f.shape[1]

        coo_data.append(W_ij)

    # Add matrix to itself to get a symmetric matrix.
    # The '+' here is list concatenation and not addition.
    # The csr_matrix constructor will do the addition later.
    ij = np.concatenate(i + j)
    ji = np.concatenate(j + i)
    coo_data = np.concatenate(coo_data + coo_data)

    # Assemble weights from coordinate sparse matrix format.
    W = scipy.sparse.csr_matrix((coo_data, (ij, ji)), (n, n))

    W = normalize_rows(W)

    I = scipy.sparse.identity(n)

    L = I - W

    return L


            
def _nncl_laplacian(image, epsilon, v_n, i_n, j_n, is_known, nn, dws=0.6):

    """
    1. Laplace-Matrix von NN-Color line Model berechnen.
    """
    h, w, _ = image.shape

    # Anzahl der Pixel
    n = h * w

    r, g, b = image.reshape(-1, 3).T
    
    # Wir bauen ein Koordinatensystem
    x_n = np.tile(np.linspace(0, 1, w), h)
    y_n = np.repeat(np.linspace(0, 1, h), w)
    
    
    # berechnen den feature vector zu allen Pixeln durch
    f = np.stack([r, g, b, x_n * dws, y_n * dws], axis=1, out=np.zeros((n, 5), dtype=np.float32))
    

    # index der nähsten nachbarn ni für alle feature vectores berechnen
    _, ni  = knn(f, f, k=nn)    #np.float32 benuzten kein 64!

    calc(image, epsilon, v_n, i_n, j_n, nn, ni, is_known)


@njit("void(f8[:, :, :], f8, i8, f8[:], i8[:], i8[:], b1[:])", cache=True, nogil=True,  parallel=True)
def _cf_laplacian(image, epsilon, radius, v_l, i_l, j_l, is_known):

    """ 
    2. Laplace-Matrix von local color line model berechnen.
    """

    h, w, _ = image.shape
    n = h*w

    size = 2 * radius + 1
    window_area = size * size


    #pixels die neighbours überhaupt haben


    # Anzahl der Pixel, welche sich angeschaut werden (manche pixel werden doppelt, 3-fach etc angeschaut),
    # da wir uns die von jedem pixel, welches nicht am Rand ist das Fenster anschauen
    
    #weil die seiten keine neighbours haben deshalb shape=(n - 2*radius*h - 2*radius*w + 2 , window_area)
    pixel_with_neighbour_count = n - 2*radius*h - 2*radius*w + 4
    neighbours_cf = np.zeros(shape=(pixel_with_neighbour_count, window_area),dtype=np.int64)


    #hier werden zunächst die lokalen nachbarn berechnet
    for y_tmp2 in numba.prange(radius, h - radius):
        for x_tmp2 in range(radius, w - radius):
            #berechnen die lokalen nachbern des pixels (y_l,x_l)
            counter = (y_tmp2-radius)*(w-radius*2) + x_tmp2 - radius  

            for y_neighbours in range(size):
                for x_neighbours in range(size):
                    counter_neighbours = y_neighbours * size + x_neighbours
                    neighbours_cf[counter, counter_neighbours] = (y_tmp2-radius + y_neighbours)*w + (x_tmp2-radius + x_neighbours)
    calc(image, epsilon, v_l, i_l, j_l,  window_area, neighbours_cf, is_known)


def lnclm_laplacian(image, is_known, epsilon=1e-7, radius=1):

    h, w, _ = image.shape

    # Anzahl der Pixel
    n = h * w
    size = 2 * radius + 1
    window_area = size * size

    n_values = (w - 2 * radius) * (h - 2 * radius) * window_area**2

    nn=9

    # inner_array  wird gespeichert y-wert
    i_n = np.zeros(((n * nn ** 2) ), dtype=np.int64)
    # welche werte genau im inner array x-wert
    j_n = np.zeros(((n * nn ** 2) ), dtype=np.int64)
    # values innerhalb dessen
    v_n = np.zeros(((n * nn ** 2) ), dtype=np.float64)
    
    # inner_array wird gespeichert y-wert
    i_l = np.zeros(n_values, dtype=np.int64)
    # welche werte genau im inner array x-wert
    j_l = np.zeros(n_values, dtype=np.int64)
    # values innerhalb dessen
    v_l = np.zeros(n_values, dtype=np.float64)


    _nncl_laplacian(image, epsilon, v_n, i_n, j_n, is_known,nn,dws=0.6)
    L_n = scipy.sparse.csr_matrix((v_n, (i_n, j_n)), shape=(n, n))
    
    _cf_laplacian(image, epsilon, radius, v_l, i_l, j_l, is_known)
    L_l = scipy.sparse.csr_matrix((v_l, (i_l, j_l)), shape=(n, n))


   
    return L_n, L_l