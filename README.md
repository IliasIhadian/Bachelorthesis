# Meine Bachelorarbeit

In dieser README.md findet ihr allgemeine Informationen zur meinem Algortihmus. Viel Spaß!

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

### Beispiel

```
python3 algo.py
Input Image and Trimap: input/mipico5x5.png input/mipico5x5_trimap.png
```

Code testen:

```
pytest
```

Benchmark für LNCLM ausrechnen:

```
python3 benchmarks.py
```

## Beschreibung

NN-Color-Line-Model Algo:

1. Zunächst wird das Image + die Trimap geladen

2. Daraufhin konstruieren wir den Feature Vector für jeden pixel

3. Dann wird KNN mit FLANN suchen

4. Es wird L_n(Laplace Matrix des NN-Color-Line-Model) sowie L_i (Laplace Matrix des local color line model) berechnet

5. der Alpha wert wird berechnet

6. dieser wird dann mit den Bild verbunden.

## Wie ist das Repository organisiert?

Das Repository beinhaltet 5 Ordner und 4 Python-files.

Im input Ordner befinden sich die die Bilder und die Trimaps, aus denen dann der Vordergrund herausgeschnitten wird(Diese sollten am besten .png Bilder seien). Die 3 Ordner im input Ordner sind von alphamatting.com.
Im latex Ordner befinden sich die wichtigen .tex dateien für meine bachelorarbeit und die ba.pdf an für sich, zudem ist dort auch eine Bilder Ordner vorzufinden, welcher die Bilder für die ba.pdf beinhaltet.
Im output Ordner befinden sich das daraus resultierende extrahierte Vordergrund und die Alpha-werte.
Im test Ordner lassen sich die Tests für die .py dateien im root Ordner.
Im letzen util Ordner sind callback.py, cg.py und kdtree.py dateien. Diese sind Hilfsdatein die im LNCLM Algorithmus benutzt werden.

Der algo.py datei besteht beinhaltet die main-funktion und greift für die Berechnung der Alpha-werte auf den estimate_alpha_lnclm.py datei. Dieser wiederum bedient sich an dem lnclm_laplacian.py datei zur Berechnung der Laplacian Matrix L_n sowie L_l.
Die benchmark.py datei rechnet den MSE und SAD für alle 27 Bilder aus alphmatting.com.

## Quellenangabe

Der Algorithmus basiert auf dem Paper:
Byoung-Kwang Kim, Meiguang Jin, and Woo-Jin Song. “Local and nonlocal color line models for image matting”. In: IEICE Transactions on Fundamentals of Electronics, Communications and Computer Sciences 97.8 (2014), pp. 1814–1819.

Das [lemur.png](https://www.flickr.com/photos/mathiasappel/25419442300/) stammt von Mathias Appel lizensiert unter [CC0 1.0 Universal (CC0 1.0) Public Domain License](https://creativecommons.org/publicdomain/zero/1.0/).
