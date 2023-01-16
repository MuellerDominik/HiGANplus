
# Handwritten Text Generation using cGAN: Sub-experiment 'Arabic Handwriting Generation'

This experiment aims to answer whether the HiGAN+ can be generalized to the Arabic language.

## Requirements

Install the packages listed in the requirement.txt file.

## Recreation of results

To run the model, first the arabic handwriting dataset AHAWP needs to be downloaded. In the dataset the sub-folder
'isolated_words_per_user'. The folder needs to be put into the sub-folder 'data' and split to 
'arabic_isolated_words_per_user_train' and 'arabic_isolated_words_per_user_test'. In our case we use 18 users 
in the test set and the rest of the users in the train set. 

Afterwards the python file named 'arabic_image_pre_processing' needs to be run. This file does get the labels,
the writer ID, crops the image according to the original IAM datset and inverts it. At the end
the 'arabic_hdf5_dataset_generation.py' is executed which generates the needed .hdf5 files to run the model. 

In order to run the code without adjusting the source code otherwise, the generated test and train sets need to be 
renamed to 'trnvalset_words64_OrgSz.hdf5' and 'testset_words64_OrgSz.hdf5' respectively. 

Furthermore a txt file named 'arabic_test_words.txt' is generated when running 

The 'reading_hdf5_files.py' was used to reverse engineer the dataset structure of the source code and for debugging
purposes only. 

The 'README_of_HiGAN+_Source_Code.md' file is the mardown README file of the source code. 