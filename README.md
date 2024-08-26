# My Bachelorthesis

In this README.md, you will find general information about my algorithm. Enjoy! U will find the the written bachelorthesis in ba.pdf.

## Getting started

### Minimal requirements

- matplotlib==3.8.0
- numba==0.59.0
- numpy==2.0.1
- opencv_python==4.9.0.80
- Pillow==10.4.0
- pyamg==5.1.0
- PyMatting==1.1.12
- Requests==2.32.3
- scikit_learn==1.2.2
- scipy==1.14.0

### Example

```
python3 algo.py
Input Image and Trimap: input/mipico5x5.png input/mipico5x5_trimap.png
```

Code testing:

```
pytest
```

Calculating benchmarks for LNCLM:

```
python3 benchmarks.py
```

## Description

NN-Color-Line-Model Algorithm:

1. First, the image and trimap are loaded.

2. Next, the feature vector is constructed for each pixel.

3. Then, KNN is searched using FLANN.

4. The Laplace matrix L_n (of the NN-Color-Line-Model) and L_i(of the local color line model) are computed.

5. The alpha value is calculated.

6. Finally, this value is combined with the image.

## How is the repository organized?

The repository contains 5 folders and 4 Python files.

The input folder contains images and trimaps from which the foreground is extracted (preferably .png images). The 3 subfolders in the input folder are from alphamatting.com.
The latex folder contains important .tex files for my bachelor thesis, the ba.pdf itself, and an images folder with the images used in ba.pdf.
The output folder contains the extracted foreground and alpha values.
The test folder includes tests for the .py files in the root directory.
The final util folder contains callback.py, cg.py, and kdtree.py files. These are utility files used in the LNCLM algorithm.

The algo.py file includes the main function and calls estimate_alpha_lnclm.py for the alpha value calculation. This, in turn, uses lnclm_laplacian.py to compute the Laplacian matrices L_n and L_l. The benchmark.py file calculates MSE and SAD for all 27 images from alphamatting.com.

## Sources

The algorithm is based on the paper:
Byoung-Kwang Kim, Meiguang Jin, and Woo-Jin Song. “Local and nonlocal color line models for image matting”. In: IEICE Transactions on Fundamentals of Electronics, Communications and Computer Sciences 97.8 (2014), pp. 1814–1819.

The [lemur.png](https://www.flickr.com/photos/mathiasappel/25419442300/) comes from Mathias Appel licensed under [CC0 1.0 Universal (CC0 1.0) Public Domain License](https://creativecommons.org/publicdomain/zero/1.0/).
