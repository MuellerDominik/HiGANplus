
# Handwritten Text Generation using cGAN: 

## Sub-experiment 'Arabic Handwriting Generation'

This experiment aims to answer whether the HiGAN+ can be generalized to the Arabic language.

### Requirements

Install the packages listed in the requirement.txt file.

### Recreation of results

To begin, download the Arabic Handwriting dataset (AHAWP) and locate the sub-folder 'isolated_words_per_user'. Move 
this folder to the 'data' sub-folder and divide it into two: 'arabic_isolated_words_per_user_train' and 
'arabic_isolated_words_per_user_test'. In this case, 18 users will be in the test set and the remaining users will be 
in the train set.

Next, run the python file 'arabic_image_pre_processing.py'. This file will gather the labels, writer ID, crop the image 
according to the IAM dataset, and invert it. Then, execute 'arabic_hdf5_dataset_generation.py' to generate the 
necessary .hdf5 files for the model.

To run the code without making any adjustments, rename the generated test and train sets to 
'trnvalset_words64_OrgSz.hdf5' and 'testset_words64_OrgSz.hdf5' respectively, and save them in the 'iam' sub-folder. 

Additionally, a file called 'arabic_test_words.txt' will be created when running the test set. This file contains the 
words that the model will be tested on. Rename it to 'english_words.txt' and save it in the 'data' folder.

The file 'reading_hdf5_files.py' was used to reverse-engineer the dataset structure and for debugging purposes only. 
Once all the above steps are completed, the model can be executed. 

The 'run.sh' file can be used to run it on the Euler cluster. 

Lastly, the 'README_of_HiGAN+_Source_Code.md' file is the markdown README file for the source code. 
