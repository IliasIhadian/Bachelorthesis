import cProfile
import io
import pstats
import time
import numpy as np
from PIL import Image
from estimate_alpha_lnclm import estimate_alpha_lnclm


def main():
    """
    Ziel: alpha ausrechnen (alles hier auf diesem file)
    """
    input_image, input_trimap = input("Input Image and Trimap: ").split()

    #input/mipico5x5.png input/mipico5x5_trimap.png 
    image = np.array(Image.open(input_image).convert("RGB")) / 255.0
    trimap = np.array(Image.open(input_trimap).convert("L")) / 255.0


    alpha = estimate_alpha_lnclm(image, trimap)

    # Wir fusionieren die Alphawerte mit den dem Bild
    cutout = np.concatenate([image, alpha[:, :, np.newaxis]], axis=2)

    """
    Nun werden die Alpha-werte als Bild abgespeichert und ausgegeben
    """

    # Hier clippen wir die werte wieder zurück zu 0-255 und konvertieren es zu uint8
    alpha = np.clip(alpha * 255, 0, 255).astype(np.uint8)

    # Hier speichern wir Alphawerte als Bild ab
    Image.fromarray(alpha).save("output/alpha.png")

    # Hier werden die Bilder geöffnet und uns gezeigt
    Image.fromarray(alpha).show(title="alpha")

    """
    Nun wird der Vordergrund als Bild abgespeichert und ausgegeben
    """

    # Hier clippen wir die werte wieder zurück zu 0-255 und konvertieren es zu uint8
    cutout = np.clip(cutout * 255, 0, 255).astype(np.uint8)

    # Hier speichern wir Alphawerte als Bild ab
    Image.fromarray(cutout).save("output/foreground.png")

    # Hier werden die Bilder geöffnet und uns gezeigt
    Image.fromarray(cutout).show(title="foreground")



if __name__ == "__main__":

    start = time.perf_counter()


    pr = cProfile.Profile()
    pr.enable()

    my_result = main()

    pr.disable()
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('tottime')
    ps.print_stats(.125)

    with open('test.txt', 'w+') as f:
        f.write(s.getvalue())


    end = time.perf_counter()
    print("time for everything: ", (end - start))
