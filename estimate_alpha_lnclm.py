import scipy.sparse.linalg
import scipy.sparse
import numpy as np
from lnclm_laplacian import lnclm_laplacian
from util.cg import *
import pyamg # type: ignore


def estimate_alpha_lnclm(image, trimap):

    h, w, _ = image.shape

    """
    3. + 4. Nun könne wir nach alpha ausrechnen  (bisher nur cf)
    """

    # Hier wird die Trimap aufgebaut
    is_fg = (trimap > 0.9).flatten()
    is_bg = (trimap < 0.1).flatten()
    is_known = is_fg | is_bg
    is_unknown = ~is_known

    L_n, L_l= lnclm_laplacian(image, is_known=is_known)

    # Hier wird die Diagonalmatrix gebaut
    d = is_known.astype(np.float64)
    D = scipy.sparse.diags(d)
    
    # Hier wird die Diagonalmatrix mit lambda multipliziert und mit LM addiert
    lambda_value = 100.0
    mui_controller = 0.5

    A = lambda_value * D + mui_controller * L_n + (1 - mui_controller) * L_l
    A = A.tocsr()

    # Hier wird b_S berechnet
    b = lambda_value * is_fg.astype(np.float64)

    #smoothed ist schneller
    ml = pyamg.smoothed_aggregation_solver(A).aspreconditioner()

    

    # Hier wird das LGS für alpha = lambda * b_S * (L + lambda * D_S)^{-1} berechnet
    alpha = cg(A, b,  M=ml)
    alpha = np.clip(alpha, 0, 1).reshape(h, w)

    return alpha