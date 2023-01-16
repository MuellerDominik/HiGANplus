import cv2
import numpy as np
import os
from os import path
from PIL import Image
from arabic_hdf5_dataset_generation import gen_h5file

# Choosing Input Directory
input_dir_name = path.join("data/arabic_isolated_words_per_user_test")
# input_dir_name = path.join("data/arabic_isolated_words_per_user_new_train")

all_wids = np.empty((0,))
all_texts = []
all_imgs = []
words_list = []

for folder in os.listdir(input_dir_name):
    print(folder)
    input_dir = os.path.join(input_dir_name, folder)
    assert path.exists(input_dir)

    for file_name in os.listdir(input_dir):
        # print("file_name", file_name)
        # image = Image.open(os.path.join(input_dir, file_name))
        image = cv2.imread(os.path.join(input_dir, file_name))

        # Desired Size of Image
        string_parts = file_name.split('_')
        # print("string_parts", string_parts)
        word_length = len(string_parts[1])
        img_width = 32 * word_length
        # print("word_length", word_length)

        # Saving the words to generate a txt file
        word = string_parts[1]
        words_list.append(word)

        # Numpy array with the writer ID's:
        writer_ID = []
        writer_ID = string_parts[0].replace("user", "")
        # print("writer_ID", writer_ID, type(writer_ID))
        writer_ID = int(writer_ID)
        all_wids = np.append(all_wids, writer_ID)
        # print("all_wids", all_wids)

        # Numpy array with label text:
        all_texts.append(string_parts[1])
        # print("all_texts", all_texts)

        # Convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Denoiseing Image
        # Apply erosion to the binary image
        # denoised = cv2.GaussianBlur(gray, (5, 5), 0)
        denoised = cv2.erode(gray, (5, 5), iterations=1)
        # print("Image after denoising:")
        # cv2.imshow("denoised", denoised)  # Display the image
        # cv2.waitKey(0)  # Wait for a key press
        # cv2.destroyAllWindows()  # Close the window

        # Threshold the image to binary
        _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        # print("Image after binarification:")
        # cv2.imshow("binary", binary)  # Display the image
        # cv2.waitKey(0)  # Wait for a key press
        # cv2.destroyAllWindows()  # Close the window

        # Find the contours of the handwriting
        # 1.Try Contours: contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        x, y, w, h = cv2.boundingRect(binary)
        cv2.rectangle(binary, (x, y), (x + w, y + h), (255, 0, 0), 2)
        # print("Image with box")
        # cv2.imshow("binary with box", binary)  # Display the image
        # cv2.waitKey(0)  # Wait for a key press
        # cv2.destroyAllWindows()  # Close the window

        # Crop the image to the bounding box
        handwriting = gray[y:y + h, x:x + w]
        # print("Image after cropping:")
        # cv2.imshow("gray image cropped", handwriting)  # Display the image
        # cv2.waitKey(0)  # Wait for a key press
        # cv2.destroyAllWindows()  # Close the window

        # Resize the images while keeping the aspect ratio the same
        # print("handwriting.shape before resizing", handwriting.shape)
        target_shape = (64, img_width)
        height, width = handwriting.shape[:2]
        original_aspect_ratio = handwriting.shape[1] / handwriting.shape[0]
        # print("shape:", handwriting.shape[1], handwriting.shape[0])

        resized_image = cv2.resize(handwriting, (int(64 * original_aspect_ratio), 64))
        # print("handwriting.shape after resizing", resized_image.shape)
        # cv2.imshow("Resized handwriting", resized_image)  # Display the image
        # cv2.waitKey(0)  # Wait for a key press
        # cv2.destroyAllWindows()  # Close the window

        # Inverting the image:
        _, resized_binary = cv2.threshold(resized_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        # cv2.imshow("Resized binary handwriting", resized_binary)  # Display the image
        # cv2.waitKey(0)  # Wait for a key press
        # cv2.destroyAllWindows()  # Close the window

        # Numpy matrix with all the pre-processed images:
        # vectorized_image = resized_binary.reshape(1, -1)
        # print("vectorized_image.shape", vectorized_image.shape)
        all_imgs.append(resized_binary)
        # print(len(all_imgs))

        print("------next_image-------")

print("end_of_dataset")

# Hdf5 File Generation Function.
#  Choose whether to generate the train or test hfd5 file here. Note that the input directory above needs
#  to be set accordingly.
gen_h5file(all_imgs, all_texts, all_wids, 'arabic_hdf5_file_train')
# gen_h5file(all_imgs, all_texts, all_wids, 'arabic_hdf5_file_test')


# Open a file for writing
with open('arabic_test_words.txt', 'w') as f:
    # Write each item in the list to the file on a new line
    for item in words_list:
        f.write(str(item) + '\n')
