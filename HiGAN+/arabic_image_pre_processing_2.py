import cv2
import numpy as np
import os
from os import path
from PIL import Image

input_dir_name = path.join("data/arabic_isolated_words_per_user")

for folder in os.listdir(input_dir_name):
    print(folder)
    input_dir = os.path.join(input_dir_name, folder)
    assert path.exists(input_dir)

    for file_name in os.listdir(input_dir):
        print("file_name", file_name)
        # image = Image.open(os.path.join(input_dir, file_name))
        image = cv2.imread(os.path.join(input_dir, file_name))

        # Desired Size of Image
        string_parts = file_name.split('_')
        word_length = len(string_parts[1])
        img_width = 32 * word_length
        print("word_length", word_length)

        # Convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Denoiseing Image
        # Apply erosion to the binary image
        # denoised = cv2.GaussianBlur(gray, (5, 5), 0)
        denoised = cv2.erode(gray, (5, 5), iterations=1)
        print("Show image after denoising:")
        cv2.imshow("denoised", denoised)  # Display the image
        cv2.waitKey(0)  # Wait for a key press
        cv2.destroyAllWindows()  # Close the window

        # Threshold the image to binary
        _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        print("Show image after binarification:")
        cv2.imshow("binary", binary)  # Display the image
        cv2.waitKey(0)  # Wait for a key press
        cv2.destroyAllWindows()  # Close the window

        # Find the contours of the handwriting
        # 1.Try Contours: contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        x, y, w, h = cv2.boundingRect(binary)
        cv2.rectangle(binary, (x, y), (x + w, y + h), (255, 0, 0), 2)
        print("drawing box on image")
        cv2.imshow("binary with box", binary)  # Display the image
        cv2.waitKey(0)  # Wait for a key press
        cv2.destroyAllWindows()  # Close the window

        # Crop the image to the bounding box
        handwriting = gray[y:y + h, x:x + w]
        print("Show image after cropping:")
        cv2.imshow("gray image cropped", handwriting)  # Display the image
        cv2.waitKey(0)  # Wait for a key press
        cv2.destroyAllWindows()  # Close the window

        # Or cut away the white pixels if shape not the same
        target_shape = (64, img_width)
        if handwriting.shape[1] < target_shape[1]:
            print("handwriting.shape[1] < target_shape[1]")
        if handwriting.shape[1] != target_shape[1]:
            handwriting = handwriting[:, :target_shape[1]]

        # Scale the image
        handwriting = cv2.resize(handwriting, target_shape, interpolation=cv2.INTER_AREA)
        print("Show after scaling")
        cv2.imshow("Handwriting", handwriting)  # Display the image
        cv2.waitKey(0)  # Wait for a key press
        cv2.destroyAllWindows()  # Close the window

        # padding the image if width not enough long
        if handwriting.shape[1] < target_shape[1]:
            padding_width = (0, target_shape[1] - handwriting.shape[1])
            handwriting = np.pad(handwriting, padding_width, mode='constant', constant_values=255)

        print("------next_image-------")
