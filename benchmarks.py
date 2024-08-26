import numpy as np
import scipy.sparse
from PIL import Image
from urllib.parse import urlparse
from numba import njit
import requests
import io
import os
from pymatting import ichol, cg, ProgressCallback, trimap_split, knn

from lnclm_laplacian import lnclm_laplacian

def test_training_set_lnclm():
    mses = []
    sads = []
    for i in range(1, 28):
        image_path = os.path.expanduser(f"~/data/rhemann/input_training_lowres/GT{i:02d}.png")
        trimap_path = os.path.expanduser(f"~/data/rhemann/trimap_training_lowres/Trimap1/GT{i:02d}.png")
        gt_alpha_path = os.path.expanduser(f"~/data/rhemann/gt_training_lowres/GT{i:02d}.png")

        image = np.array(Image.open(image_path).convert("RGB")) / 255.0
        trimap = np.array(Image.open(trimap_path).convert("L")) / 255.0
        gt_alpha = np.array(Image.open(gt_alpha_path).convert("L")) / 255.0

        alpha = estimate_alpha_lnclm(image, trimap, gt_alpha)

        _, _, _, is_unknown = trimap_split(trimap)

        mse = compute_mse(alpha, gt_alpha, is_unknown.reshape(alpha.shape))
        sad = np.sum(np.abs(alpha - gt_alpha)[is_unknown.reshape(alpha.shape)])

        print(f"MSE: {mse:.10f}, SAD: {sad:16.10f}, image: {i:02d}")

        mses.append(mse)
        sads.append(sad)

    print(f"Mean MSE: {np.mean(mses):.10f}")
    print(f"Mean SAD: {np.mean(sads):.10f}")
    print()

def make_knn_neighbors(image, k=9):
    h, w = image.shape[:2]
    n = h * w
    r, g, b = image.reshape(n, 3).T
    y, x = np.mgrid[:h, :w].reshape(2, n)
    scale = 1 / np.hypot(w, h)

    features = np.stack([r, g, b, scale * x, scale * y], axis=1).astype(np.float32)
    return knn(features, features, k=k)[1].astype(np.int32).reshape(h, w, k)

def estimate_alpha_lnclm(image, trimap, gt_alpha, epsilon=1e-7, k=9, lambd=100.0, u=0.5, r=1):
    h, w = image.shape[:2]

    is_fg, _, is_known, is_unknown = trimap_split(trimap)

    neighbors_cf = make_neighbors(h, w, r)
    neighbors_nn = make_knn_neighbors(image, k=k)

    is_known_laplacian = is_known#np.zeros_like(is_known)

    """ L_cf = laplacian(image, epsilon, is_known_laplacian, neighbors_cf)
    L_nn = laplacian(image, epsilon, is_known_laplacian, neighbors_nn)
 """
    L_nn, L_cf = lnclm_laplacian(image, is_known_laplacian)

    A = u * L_nn + (1.0 - u) * L_cf + scipy.sparse.diags(lambd * is_known)
    b = lambd * is_fg

    import pyamg
    M = pyamg.smoothed_aggregation_solver(A).aspreconditioner()
    #M = ichol(A)

    x = cg(A, b, M=M, rtol=0, atol=1e-10)#, callback=ProgressCallback())

    alpha = np.clip(x, 0, 1).reshape(h, w)

    if 0:
        import matplotlib.pyplot as plt
        plt.imshow(alpha, cmap="gray")
        plt.show()

    return alpha

def test_training_set_cf():
    mses = []
    sads = []
    for i in range(1, 28):
        image_path = os.path.expanduser(f"~/data/rhemann/input_training_lowres/GT{i:02d}.png")
        trimap_path = os.path.expanduser(f"~/data/rhemann/trimap_training_lowres/Trimap1/GT{i:02d}.png")
        gt_alpha_path = os.path.expanduser(f"~/data/rhemann/gt_training_lowres/GT{i:02d}.png")

        image = np.array(Image.open(image_path).convert("RGB")) / 255.0
        trimap = np.array(Image.open(trimap_path).convert("L")) / 255.0
        gt_alpha = np.array(Image.open(gt_alpha_path).convert("L")) / 255.0

        alpha = estimate_alpha_cf(image, trimap)

        _, _, _, is_unknown = trimap_split(trimap)

        mse = compute_mse(alpha, gt_alpha, is_unknown.reshape(alpha.shape))
        sad = np.sum(np.abs(alpha - gt_alpha)[is_unknown.reshape(alpha.shape)])

        mses.append(mse)
        sads.append(sad)

        print(f"MSE: {mse:.10f}, SAD: {sad:16.10f}, image: {i:02d}")

    print(f"Mean MSE: {np.mean(mses):.10f}")
    print(f"Mean SAD: {np.mean(sads):.10f}")

def test_alphamatting_com_test_images():
    names = ["troll", "doll", "donkey", "elephant", "plant", "pineapple", "plasticbag", "net"]

    for name in names:
        test_image(name)

def main():
    # Same result as Table 2, column 1 of paper
    # "Local and Nonlocal Color Line Models for Image Matting"
    # Mean MSE: 0.0218078909
    # Mean SAD: 5310.9017503770
    #test_training_set_cf()

    # Error slightly too large (should be 0.0175 MSE and 4750 SAD)
    # Mean MSE: 0.0185104887
    # Mean SAD: 4823.9560169277
    test_training_set_lnclm()

    # Very small error for every image
    """
    MSE: 0.0000022161 max difference: 0.0039215686, image: troll
    MSE: 0.0000052303 max difference: 0.0039215686, image: doll
    MSE: 0.0000029910 max difference: 0.0039215686, image: donkey
    MSE: 0.0000042086 max difference: 0.0039215686, image: elephant
    MSE: 0.0000026321 max difference: 0.0039215686, image: plant
    MSE: 0.0000067701 max difference: 0.0078431373, image: pineapple
    MSE: 0.0000036779 max difference: 0.0039215686, image: plasticbag
    MSE: 0.0000031328 max difference: 0.0039215686, image: net
    """
    #test_alphamatting_com_test_images()

def test_image(name):
    expected_alpha_url = f"http://www.alphamatting.com/uploaded/222/standardtest/Trimap1/{name}.png"
    trimap_url = f"http://alphamatting.com/datasets/Trimap1/{name}.png"
    image_url = f"http://alphamatting.com/datasets/testimages/{name}.png"

    expected_alpha = np.array(Image.open(io.BytesIO(download_cached(expected_alpha_url))).convert("L")) / 255.0
    trimap = np.array(Image.open(io.BytesIO(download_cached(trimap_url))).convert("L")) / 255.0
    is_unknown = (trimap != 0.0) & (trimap != 1.0)
    image = np.array(Image.open(io.BytesIO(download_cached(image_url))).convert("RGB")) / 255.0

    alpha = estimate_alpha_cf(image, trimap)

    alpha = (alpha * 255 + 0.5).astype(np.uint8) / 255.0

    mse = 10.0 * compute_mse(alpha, expected_alpha, is_unknown)

    difference = np.abs(expected_alpha - alpha)

    print(f"MSE: {mse:.10f} max difference: {difference.max():.10f}, image: {name}")

    import matplotlib.pyplot as plt
    for i, img in enumerate([image, expected_alpha, alpha, difference], 1):
        plt.subplot(2, 2, i)
        plt.imshow(img, cmap="gray", vmin=0, vmax=1)
        plt.axis("off")
    plt.show()

@njit("Tuple((i4[:], i4[:], f8[:]))(f8[:, :, ::1], f8, b1[::1], i4[:, :, :])", cache=True, nogil=True)
def laplacian_entrie(image, epsilon, is_known, neighbors):
    h, w, d = image.shape
    n = h * w
    n_neighbors = neighbors.shape[2]
    image = image.reshape(n, d)
    i_inds = np.zeros((h, w, n_neighbors, n_neighbors), dtype=np.int32)
    j_inds = np.zeros((h, w, n_neighbors, n_neighbors), dtype=np.int32)
    values = np.zeros((h, w, n_neighbors, n_neighbors), dtype=np.float64)

    # For each pixel of image
    for y in range(h):
        for x in range(w):
            # Skip known pixels
            if np.all(is_known[neighbors[y, x]]): continue

            # Compute inverse covariance over neighborhood
            mean_colors = image[neighbors[y, x], :].sum(axis=0).reshape(1, 3)
            c = image[neighbors[y, x], :] - mean_colors / n_neighbors
            inv_cov = np.linalg.inv(epsilon * np.eye(3) + c.T @ c) * n_neighbors

            # For each pair (i, j) in neighborhood
            for di in range(n_neighbors):
                i = neighbors[y, x, di]
                u = inv_cov @ c[di]

                for dj in range(n_neighbors):
                    j = neighbors[y, x, dj]

                    # Calculate contribution of pixel pair to L_ij
                    value = (i == j) - (1 + np.dot(u, c[dj])) / n_neighbors

                    i_inds[y, x, di, dj] = i
                    j_inds[y, x, di, dj] = j
                    values[y, x, di, dj] = value

    return i_inds.ravel(), j_inds.ravel(), values.ravel()

@njit("i4[:, :, :](i4, i4, i4)", cache=True)
def make_neighbors(h, w, r):
    neighbors = np.zeros((h, w, (2 * r + 1)**2), dtype=np.int32)
    for y in range(r, h - r):
        for x in range(r, w - r):
            k = 0
            for dy in range(2 * r + 1):
                for dx in range(2 * r + 1):
                    x2 = x + dx - 1
                    y2 = y + dy - 1
                    neighbors[y, x, k] = x2 + y2 * w
                    k += 1
    return neighbors
""" 
def laplacian(image, epsilon, is_known, neighbors):
    h, w = image.shape[:2]
    i_inds, j_inds, values = laplacian_entrie(image, epsilon, is_known, neighbors)
    n = h * w
    return scipy.sparse.csr_matrix((values, (i_inds, j_inds)), (n, n)) """

def estimate_alpha_cf(image, trimap, epsilon=1e-7, r=1):
    h, w = trimap.shape[:2]

    is_fg, _, is_known, is_unknown = trimap_split(trimap)

    neighbors = make_neighbors(h, w, r)

    L = lnclm_laplacian(image, is_known)

    L_U = L[is_unknown, :][:, is_unknown]

    R = L[is_unknown, :][:, is_known]

    m = is_fg[is_known]

    x = trimap.copy().ravel()

    M = ichol(L_U)

    x[is_unknown] = cg(L_U, -R.dot(m), M=M, callback=ProgressCallback(), maxiter=30)

    alpha = np.clip(x, 0, 1).reshape(h, w)

    return alpha

def compute_mse(image, true_image, is_unknown):
    return np.mean(np.square(image - true_image)[is_unknown])

def download_cached(url):
    path = urlparse(url).path[1:]

    # Create directory if necessary
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    # Load bytes from file if it exists
    if os.path.isfile(path):
        with open(path, "rb") as f:
            return f.read()

    # Download bytes from URL
    with requests.get(url) as response:
        response.raise_for_status()
        data = response.content
        assert len(data) > 0

    # Save bytes to file
    with open(path, "wb") as f:
        f.write(data)

    return data

if __name__ == "__main__":
    main()
