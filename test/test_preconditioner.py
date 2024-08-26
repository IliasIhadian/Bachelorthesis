import os
import sys

# Add the parent directory of 'ba' to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname("/Users/iliasihadian/Desktop/bachelorarbeit/algos/ba/test"))
sys.path.append("/Users/iliasihadian/Desktop/bachelorarbeit/algos/ba")


import numpy as np
from PIL import Image
from util.cg import *
from util.callback import CounterCallback
import scipy
from lnclm_laplacian import *
import pyamg # type: ignore



def test_preconditioners():
    image = np.array(Image.open("input/lemur.png").convert("RGB")) / 255.0
    trimap = np.array(Image.open("input/lemur_trimap.png").convert("L")) / 255.0


    # Hier wird die Trimap aufgebaut
    is_fg = (trimap > 0.9).flatten()
    is_bg = (trimap < 0.1).flatten()
    is_known = is_fg | is_bg

    L_n, L_l= lnclm_laplacian(image, is_known=is_known);
    # Hier wird die Diagonalmatrix gebaut
    d = is_known.astype(np.float64)
    D = scipy.sparse.diags(d)
    
    # Hier wird die Diagonalmatrix mit lambda multipliziert und mit LM addiert
    lambda_value = 100.0
    mui_controller = 0.5

    A = lambda_value * D + mui_controller * L_n + (1 - mui_controller) * L_l
    A = A.tocsr()

    b = lambda_value * is_fg.astype(np.float64)

    ml = pyamg.smoothed_aggregation_solver(A).aspreconditioner()                    # construct the multigrid hierarchy

    callback = CounterCallback()
    
    alpha = cg(A, b,  M=ml, callback=callback)

    expected_iterations = 29

    if callback.n > expected_iterations:
        print(
            "WARNING: Unexpected number of iterations. Expected %d, but got %d"
            % (expected_iterations, callback.n)
        )
    assert callback.n <= 40