from PIL import Image
from matplotlib import pyplot as plt
from skimage import measure
import cv2 as cv
import numpy as np
import sys
import os


def main():
    img_path = sys.argv[1]
    func_string = sys.argv[2]
    save_location = sys.argv[3]

    img = cv.imread(img_path)
    if img is None:
        print('Could not open or find the image')
        exit(0)

    func_list = list(func_string.split(","))

    func_dict = {
        "CS": contrast_stretching,
        "HE": histogram_equalize,
        "CL": clahe,
        "GC": gamma_correction,
        "BC": apply_brightness_contrast,
        "NR": erosion_dilation,
        "UM": unsharp_masking,
    }

    new_image = img
    for i in range(len(func_list)):
        func = func_dict[func_list[i]]
        new_image = func(new_image)

    cv.imshow('Original', img)
    cv.imshow(func_string, new_image)
    cv.imwrite(os.path.join(save_location, func_string + ".jpg"), new_image)

    cv.waitKey(0)
    cv.destroyAllWindows()


def test_dataset(data_set, functions):
    average_improvement = 0
    average_original = 0
    average_new = 0
    counter = 0

    original_path = "OneClic/" + data_set
    oneclick_path = "OneClic/" + data_set + "_usingOneClick"
    destination_path = "OneClic/" + data_set + "_usingTwoClick"

    functions_string = ""
    for i in range(len(functions)):
        functions_string += functions[i].__name__ + "_"

    destination_path = os.path.join(destination_path, functions_string)

    os.makedirs(destination_path, exist_ok=True)

    csv_name = functions_string + data_set + ".csv"
    csv_name = os.path.join(destination_path, csv_name)
    f = open(csv_name, "w")
    f.write("Filename, Original_Similarity, New_Similarity, Improvement\n")

    for filename in os.listdir(original_path):
        # loads images
        img = cv.imread(os.path.join(original_path, filename))
        one_click = cv.imread(os.path.join(oneclick_path, filename))

        if img is None or one_click is None:
            print('Could not open or find image')
            continue

        # resize image to enable comparison
        img = cv.resize(img, (int(one_click.shape[1]), int(one_click.shape[0])))

        # generates new image
        new_image = img
        for i in range(len(functions)):
            func = functions[i]
            new_image = func(new_image)

        # https://www.pyimagesearch.com/2014/09/15/python-compare-two-images/
        my_percentage = measure.compare_ssim(new_image, one_click, multichannel=True)
        average_new += my_percentage
        # compute likeness percentage between original and new image
        # my_dif = cv.absdiff(new_image, one_click)
        # my_dif = my_dif.astype(np.uint8)
        # my_percentage = (np.count_nonzero(my_dif) * 100) / my_dif.size

        their_percentage = measure.compare_ssim(img, one_click, multichannel=True)
        average_original += their_percentage
        # compute likeness percentage between original and oneClick
        # their_dif = cv.absdiff(img, one_click)
        # their_dif = their_dif.astype(np.uint8)
        # their_percentage = (np.count_nonzero(their_dif) * 100) / their_dif.size

        # print similarity and improvement to file
        f.write(filename + ", " + str(their_percentage) + ", " + str(my_percentage) + ", " + str(
            my_percentage - their_percentage) + "\n")
        average_improvement += my_percentage - their_percentage
        counter += 1

        # store difference image
        difference = cv.subtract(one_click, new_image)
        cv.imwrite(os.path.join(destination_path, "dif_" + filename), difference)

        # stores new image
        cv.imwrite(os.path.join(destination_path, filename), new_image)

    f.write("AVERAGES, " + str(average_original) + ", " + str(average_new) + ", " + str(average_improvement / counter) + "\n")
    f.close()


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


def erosion_dilation(img):
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


if __name__ == "__main__":
    # function_set = [contrast_stretching, clahe, histogram_equalize, gamma_correction, apply_brightness_contrast, erosion_dilation, unsharp_masking]
    # for i in range(len(function_set)):
    #    test_dataset("good", [function_set[i]])

    main()
