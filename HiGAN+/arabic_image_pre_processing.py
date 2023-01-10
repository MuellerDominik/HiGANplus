# Crop image from center
# remove whitespace
# convert to grayscale
# resize to 64X64

import cv2
import os
from os import path
from PIL import Image
from PIL import ImageOps
import imutils


def crop_center(pil_img, crop_width, crop_height):
    img_width, img_height = pil_img.size
    return pil_img.crop(((img_width - crop_width) // 2,
                         (img_height - crop_height) // 2,
                         (img_width + crop_width) // 2,
                         (img_height + crop_height) // 2))


input_dir_name = path.join("data/arabic_isolated_words_per_user")

for folder in os.listdir(input_dir_name):
    print(folder)
    input_dir = os.path.join(input_dir_name, folder)
    assert path.exists(input_dir)

    for file_name in os.listdir(input_dir):
        print("file_name", file_name)
        image = Image.open(os.path.join(input_dir, file_name))
        image.load()
        imageSize = image.size
        print("Showing Original Images:")
        image.show()

        # Resizing Image
        string_parts = file_name.split('_')
        word_length = len(string_parts[1])
        print("word_length", word_length)
        img_width = 16 * word_length
        resized_image = imutils.resize(image, width=img_width, height=32)
        # resized_image = crop_center(image, 16*word_length, imageSize[0])

        # Inversion
        # invert_im = resized_image.convert("RGB")
        # print("Showing Inverted Images:")
        # invert_im.show()
        # imageBox = invert_im.getbbox()

        imageBox = image.getbbox()
        cropped_image = image.crop(imageBox)
        print("%s Size:%s New Size:%s" % (file_name, imageSize, imageBox))
        cropped_image.save(os.path.join(input_dir, file_name))

    # For every file in the input dir and resize to [32 * 16*len(text)]
    for file in os.listdir(input_dir):
        original_image = cv2.imread(os.path.join(input_dir, file))
        cv2.imshow('original', original_image)
        # resize image with aspect ratio maintained
        print("file", file)
        img_width = 16 * len(file)  # TODO: Need the length of the arabic word in the file name
        resized_img = imutils.resize(original_image, width=img_width, height=32)
        # resized_img = cv2.resize(original_image, (128, 128), cv2.INTER_AREA)
        cv2.imshow('resized', resized_img)
        cv2.imwrite(os.path.join(input_dir, file), resized_img)
