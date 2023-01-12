import cv2
import os
from os import path
from PIL import Image
from PIL import ImageOps
import imutils

input_dir_name = path.join("data/arabic_isolated_words_per_user")

for folder in os.listdir(input_dir_name):
    print(folder)
    input_dir = os.path.join(input_dir_name, folder)
    assert path.exists(input_dir)

    for file_name in os.listdir(input_dir):
        print("file_name", file_name)
        image = Image.open(os.path.join(input_dir, file_name))
        image.load()
        image.show()

        # Resizing Image
        string_parts = file_name.split('_')
        word_length = len(string_parts[1])
        img_width = 16 * word_length
        print("word_length", word_length)

        # Get the original width and height
        imageSize = image.size
        width, height = image.size

        # Calculate the new width and height
        new_height = 64
        new_width = 16 * word_length

        # Height adjust the image
        image = image.resize((new_width, new_height), resample=Image.LANCZOS)
        print("Show 1")
        image.show()

        # Crop or pad the image
        if width > new_width:
            # Crop the image by slicing off the excess on the right
            image = image.crop((0, 0, new_width, new_height))
        elif width < new_width:
            # Pad the image by creating a new image with the desired size and pasting the original image in the center
            padded_im = Image.new("RGB", (new_width, new_height), (255, 255, 255))
            padded_im.paste(image, ((new_width - width) // 2, (new_height - height) // 2))
            image = padded_im

        print("Show 2")
        image.show()

        # Save the modified image
        imageNewSize = image.size
        image.save("image_cropped_or_padded.png")
        # resized_image.save(os.path.join(input_dir, file_name))

        print("%s Size:%s New Size:%s" % (file_name, imageSize, imageNewSize))

    # # For every file in the input dir and resize to [32 * 16*len(text)]
    # for file in os.listdir(input_dir):
    #     original_image = cv2.imread(os.path.join(input_dir, file))
    #     cv2.imshow('original', original_image)
    #     # resize image with aspect ratio maintained
    #     print("file", file)
    #     img_width = 16 * len(file)  # Need the length of the arabic word in the file name
    #     resized_img = imutils.resize(original_image, width=img_width, height=32)
    #     # resized_img = cv2.resize(original_image, (128, 128), cv2.INTER_AREA)
    #     cv2.imshow('resized', resized_img)
    #     cv2.imwrite(os.path.join(input_dir, file), resized_img)
