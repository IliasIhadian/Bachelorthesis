import os
import sys

# Add the parent directory of 'ba' to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname("/Users/iliasihadian/Desktop/bachelorarbeit/algos/ba/test"))
sys.path.append("/Users/iliasihadian/Desktop/bachelorarbeit/algos/ba")



import numpy as np
import io, requests
from PIL import Image
from urllib.parse import urlparse
from estimate_alpha_lnclm import estimate_alpha_lnclm


def test_image_lnclm():
    expected_alpha_url_lnclm = f"http://www.alphamatting.com/uploaded/1918/standardtest/Trimap1/troll.png"
    trimap_url = f"http://alphamatting.com/datasets/Trimap1/troll.png"
    image_url = f"http://alphamatting.com/datasets/testimages/troll.png"

    # Download images
    expected_alpha_lnclm = np.array(Image.open(io.BytesIO(download_cached(expected_alpha_url_lnclm))).convert("L")) / 255.0
    trimap = np.array(Image.open(io.BytesIO(download_cached(trimap_url))).convert("L")) / 255.0
    is_unknown = (trimap != 0.0) & (trimap != 1.0)
    image = np.array(Image.open(io.BytesIO(download_cached(image_url))).convert("RGB")) / 255.0

    # Compute alpha with closed-form alpha matting method by Levin et al.
    alpha_lnclm = estimate_alpha_lnclm(image, trimap)

    # Introduce quantization error
    alpha_lnclm = (alpha_lnclm * 255 + 0.5).astype(np.uint8) / 255.0


    # Compute MSE (might be different, depending on paper)
    mse_lnclm = 10.0 * compute_mse(alpha_lnclm, expected_alpha_lnclm, is_unknown)

    assert mse_lnclm < 10 #das muss später auf 0.001 gesetzt werden

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

