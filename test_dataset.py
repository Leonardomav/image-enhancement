from skimage import measure
import cv2 as cv
import os


def test_dataset(data_set, functions):
    average_improvement = 0
    average_original = 0
    average_new = 0
    counter = 0

    original_path = "OneClic/" + data_set
    oneclick_path = "OneClic/" + data_set + "_usingOneClick"
    destination_path = "OneClic/" + data_set + "_results"

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

        # https://www.cns.nyu.edu/pub/eero/wang03-reprint.pdf
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
