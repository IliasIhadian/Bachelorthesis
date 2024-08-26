import os
import sys


# Add the parent directory of 'ba' to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname("/Users/iliasihadian/Desktop/bachelorarbeit/algos/ba/test"))
sys.path.append("/Users/iliasihadian/Desktop/bachelorarbeit/algos/ba")


import numpy as np
from PIL import Image
from estimate_alpha_lnclm import estimate_alpha_lnclm


def test_alpha():

    image = np.array(Image.open("input/input_training_lowres/GT01.png").convert("RGB")) / 255.0
    trimap = np.array(Image.open("input/trimap_training_lowres/Trimap1/GT01.png").convert("L")) / 255.0
    true_alpha = np.array(Image.open("input/gt_training_lowres/GT01.png").convert("L")) / 255.0


    alpha = estimate_alpha_lnclm(image, trimap)

    error = np.linalg.norm(alpha - true_alpha)

    max_error = 12
    print(error)

    assert error < max_error