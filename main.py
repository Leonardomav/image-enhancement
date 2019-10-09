from skimage import measure
import cv2 as cv
import sys
import os
import classic_enhancement


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
        "CS": classic_enhancement.contrast_stretching,
        "HE": classic_enhancement.histogram_equalize,
        "CL": classic_enhancement.clahe,
        "GC": classic_enhancement.gamma_correction,
        "BC": classic_enhancement.apply_brightness_contrast,
        "FD": classic_enhancement.fast_n1_denoising,
        "UM": classic_enhancement.unsharp_masking,
    }

    new_image = img
    for i in range(len(func_list)):
        try:
            func = func_dict[func_list[i]]
        except KeyError as e:
            print("Function not available")
            exit(0)
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


        their_percentage = measure.compare_ssim(img, one_click, multichannel=True)
        average_original += their_percentage


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


if __name__ == "__main__":
    # function_set = [classic_enhancement.contrast_stretching, classic_enhancement.clahe, classic_enhancement.histogram_equalize,
    #                 classic_enhancement.gamma_correction, classic_enhancement.apply_brightness_contrast,
    #                 classic_enhancement.fast_n1_denoising, classic_enhancement.unsharp_masking]
    # for i in range(len(function_set)):
    #    test_dataset("good", [function_set[i]])

    main()
