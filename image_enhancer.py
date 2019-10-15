from PIL import Image
from matplotlib import pyplot as plt
from skimage import measure
import cv2 as cv
import sys
import os
import numpy as np
import test_dataset


def main():
    if len(sys.argv) < 3:
        print("Incorrect use of the scrip.\nimage_enhancer.py <input_image_path> <list_of_methods> <optional_save_path>")
        exit(0)

    func_string = sys.argv[2]
    func_list = list(func_string.split(","))

    func_dict = {
        "CS": contrast_stretching,
        "HE": histogram_equalize,
        "CL": clahe,
        "GC": gamma_correction,
        "NL": non_local_means_denoising,
        "UM": unsharp_masking,
    }

    if sys.argv[1] == "-TEST":
        real_func_list = []
        for i in range(len(func_list)):
            try:
                real_func_list.append(func_dict[func_list[i]])
            except KeyError as e:
                print("Function not available")
                exit(0)
        if len(sys.argv)==3 or sys.argv[3] == "all":
            test_dataset.test_dataset("bad", real_func_list)
            test_dataset.test_dataset("medium", real_func_list)
            test_dataset.test_dataset("good", real_func_list)
        else:
            test_dataset.test_dataset(sys.argv[3], real_func_list)
        exit(0)

    img_path = sys.argv[1]

    try:
        save_location = sys.argv[3]
    except IndexError:
        save_location = None

    img = cv.imread(img_path)
    if img is None:
        print('Could not open or find the image')
        exit(0)

    new_image = img
    for i in range(len(func_list)):
        try:
            func = func_dict[func_list[i]]
        except KeyError as e:
            print("Function not available")
            exit(0)
        new_image = func(new_image)

    save_name = os.path.basename(img_path)
    save_name = save_name.split(".")[0]
    save_name =func_string+"_"+save_name+".png"
    if save_location is not None:
        cv.imwrite(os.path.join(save_location, save_name), new_image)

    cv.waitKey(0)
    cv.destroyAllWindows()


# considers the global contrast value of an image and affects all the pixels in the same way
def histogram_equalize(img):
    b, g, r = cv.split(img)
    red = cv.equalizeHist(r)
    green = cv.equalizeHist(g)
    blue = cv.equalizeHist(b)
    out = cv.merge((blue, green, red))
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
    return out


# https://en.wikipedia.org/wiki/Unsharp_masking
def unsharp_masking(img):
    gaussian = cv.GaussianBlur(img, (5, 5), 0, 0)
    unsharp_image = cv.addWeighted(img, 1.4, gaussian, -0.4, 0)
    return unsharp_image


# https://www.pyimagesearch.com/2015/10/05/opencv-gamma-correction/
# https://en.wikipedia.org/wiki/Gamma_correction
def gamma_correction(img, gamma=1.3):
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv.LUT(img, table)


# http://www.ipol.im/pub/art/2011/bcm_nlm/
def non_local_means_denoising(img):
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    out = cv.fastNlMeansDenoisingColored(img, None, 2, 2, 7, 21)
    out = cv.cvtColor(out, cv.COLOR_RGB2BGR)
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
    return out


if __name__ == "__main__":
    main()
