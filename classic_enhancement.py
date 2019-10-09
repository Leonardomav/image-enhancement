from PIL import Image
from matplotlib import pyplot as plt
import cv2 as cv
import numpy as np


# considers the global contrast value of an image and affects all the pixels in the same way
def histogram_equalize(img):
    b, g, r = cv.split(img)
    red = cv.equalizeHist(r)
    green = cv.equalizeHist(g)
    blue = cv.equalizeHist(b)
    out = cv.merge((blue, green, red))
    # cv.imshow('Histogram equalized', out)
    return out


# https://theailearner.com/2019/04/14/adaptive-histogram-equalization-ahe/
# https://en.wikipedia.org/wiki/Adaptive_histogram_equalization
# Contrast Limited Adaptive Histogram Equalization
def clahe(img):
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    h, s, v = cv.split(hsv)
    clahe = cv.createCLAHE(clipLimit=1.2, tileGridSize=(8, 8))
    new_v = clahe.apply(v)
    hsv = cv.merge((h, s, new_v))
    out = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
    # cv.imshow('CLAHE', out)
    return out


# https://en.wikipedia.org/wiki/Unsharp_masking
def unsharp_masking(img):
    gaussian = cv.GaussianBlur(img, (5, 5), 0, 0)
    unsharp_image = cv.addWeighted(img, 1.4, gaussian, -0.4, 0)
    # cv.imshow('unsharp_masking', unsharp_image)
    return unsharp_image


# https://www.pyimagesearch.com/2015/10/05/opencv-gamma-correction/
# https://en.wikipedia.org/wiki/Gamma_correction
def gamma_correction(img, gamma=1.3):
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    # cv.imshow("Gamma Corrected", cv.LUT(img, table))
    return cv.LUT(img, table)


# https://stackoverflow.com/questions/39308030/how-do-i-increase-the-contrast-of-an-image-in-python-opencv
def apply_brightness_contrast(input_img, brightness=0, contrast=22):
    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow) / 255
        gamma_b = shadow
        buf = cv.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
    else:
        buf = input_img.copy()

    if contrast != 0:
        f = 131 * (contrast + 127) / (127 * (131 - contrast))
        alpha_c = f
        gamma_c = 127 * (1 - f)

        buf = cv.addWeighted(buf, alpha_c, buf, 0, gamma_c)

    # cv.imshow("apply_brightness_contrast", buf)
    return buf


# http://www.ipol.im/pub/art/2011/bcm_nlm/
def fast_n1_denoising(img):
    # switch to rgb
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    out = cv.fastNlMeansDenoisingColored(img, None, 2, 2, 7, 21)
    out = cv.cvtColor(out, cv.COLOR_RGB2BGR)
    # cv.imshow("Denoised", out)
    return out


# https://homepages.inf.ed.ac.uk/rbf/HIPR2/stretch.htm
def contrast_stretching(img):
    # plt.hist(img.ravel(), 256, [0, 256])
    # plt.show()
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img)
    r, g, b = pil_img.split()

    min_b = np.min(b)
    max_b = np.max(b)
    min_g = np.min(g)
    max_g = np.max(g)
    min_r = np.min(r)
    max_r = np.max(r)

    norm_b = b.point(lambda i: (i - min_b) * ((255 - 0) / (max_b - min_b)))
    norm_g = g.point(lambda i: (i - min_g) * ((255 - 0) / (max_g - min_g)))
    norm_r = r.point(lambda i: (i - min_r) * ((255 - 0) / (max_r - min_r)))

    out = Image.merge("RGB", (norm_r, norm_g, norm_b))

    out = np.array(out)
    # Convert RGB to BGR
    out = out[:, :, ::-1].copy()

    # plt.hist(out.ravel(), 256, [0, 256])
    # plt.show()
    # cv.imshow('Contrast Stretching', out)
    return out
