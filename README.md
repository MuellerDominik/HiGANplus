
# Handwritten Text Generation using a cGAN

## Introduction

This is a fork of the [HiGANplus](https://github.com/ganji15/HiGANplus) repository for the ETH course 263-3210-00L Deep Learning 2022.

## Installation

The current version has been tested on the [ETH Euler cluster](https://scicomp.ethz.ch/wiki/Euler).

Clone this repository into your home (`~`) directory with the following command:
```sh
git clone https://github.com/MuellerDominik/HiGANplus.git
```

To install the required Python packages use the following command:
```sh
pip3 install --user -r ~/HiGANplus/requirements.txt
```

To train on the [IAM dataset](https://fki.tic.heia-fr.ch/databases/iam-handwriting-database), the following two files need to be added to the directory `~/HiGANplus/HiGAN+/data/iam/`:

- [trnvalset_words64_OrgSz.hdf5](https://github.com/ganji15/HiGANplus/releases/download/dataset/trnvalset_words64_OrgSz.hdf5)
- [testset_words64_OrgSz.hdf5](https://github.com/ganji15/HiGANplus/releases/download/dataset/testset_words64_OrgSz.hdf5)

## Training

To run the training on the Euler cluster, the `NETZH_USR` variable in the shell script `~/HiGANplus/run.sh` needs to be updated.

The training batch job can be submitted with the following command:
```sh
sbatch --ntasks=1 --cpus-per-task=8 --mem-per-cpu=4G --gpus=titan_rtx:1 --time=24:00:00 ~/HiGANplus/run.sh
```

## Modifications

<!-- Patch Discriminator Adjustment -->

### Patch Discriminator Adjustment

Modifying the hyperparameters of the patch discriminator requires a modification of the Python file `~/HiGANplus/HiGAN+/networks/utils.py`. The function signature `extract_all_patches` on line `308` can be modified in the following ways:

- Changing the parameter `block_size` results in a different patch size
- Changing the parameter `step` results in a different stride and therefore controls the patch overlap

The `block_size` and `step` parameters are both used for the horizontal and vertical dimensions respectively. Changing this behaviour would require to change the function `extract_all_patches`.

<!-- Arabic Handwriting Generation -->

### Arabic Handwriting Generation

This experiment aims to answer, whether the HiGAN+ model is able to generalize to the Arabic language.

#### Installation

To train on the [AHAWP dataset](https://data.mendeley.com/datasets/2h76672znt/1), replace the following three files in the directories `~/HiGANplus/HiGAN+/data/` and `~/HiGANplus/HiGAN+/data/iam/` and refer to the section [Training](#training):

- [trnvalset_words64_OrgSz.hdf5](https://github.com/MuellerDominik/HiGANplus/releases/download/ahawp-dataset/trnvalset_words64_OrgSz.hdf5)
- [testset_words64_OrgSz.hdf5](https://github.com/MuellerDominik/HiGANplus/releases/download/ahawp-dataset/testset_words64_OrgSz.hdf5)
- [english_words.txt](https://github.com/MuellerDominik/HiGANplus/releases/download/ahawp-dataset/english_words.txt)

To generate these files from scratch, see [here](#dataset-generation).

#### Dataset Generation

First, download the [AHAWP dataset](https://data.mendeley.com/datasets/2h76672znt/1) and locate the directory `isolated_words_per_user`. Divide this directory into `arabic_isolated_words_per_user_train` and `arabic_isolated_words_per_user_test` and move them to the directory `~/HiGANplus/Arabic/data`. In this case, 18 users will be in the test split and the remaining users will be in the training split.

To generate the `.hdf5` and `.txt` files necessary for the training, run the Python file `~/HiGANplus/Arabic/arabic_image_pre_processing.py`. This file will gather the labels, writer ID, crop the image according to the IAM dataset, and invert it. The `.hdf5` files are created with the function `gen_h5file` from the Python file `~/HiGANplus/Arabic/arabic_hdf5_dataset_generation.py` which was created by the [HiGAN authors](https://github.com/ganji15/HiGAN/issues/3).

To run the code without making any adjustments, rename the generated test and train splits to `testset_words64_OrgSz.hdf5` and `trnvalset_words64_OrgSz.hdf5` respectively, and move them to the directory `~/HiGANplus/HiGAN+/data/iam/`. Additionally, a file called `arabic_test_words.txt` will be created when creating the test split (`is_test = True`). This file contains the words that the model will be tested on. Rename it to `english_words.txt` and move it to the directory `~/HiGANplus/HiGAN+/data/`.

The file `~/HiGANplus/Arabic/reading_hdf5_files.py` was used to reverse-engineer the dataset structure and for debugging purposes only. Once all the above steps are completed, the model can be trained (refer to the section [Training](#training))

<!-- Conditional GAN Modification -->

### Conditional GAN Modification

To recreate the results of the cGAN modification, clone the original [HiGANplus](https://github.com/ganji15/HiGANplus) repository and carry out the following steps:

- Replace the file `HiGANplus/HiGAN+/networks/model.py` with the file `~/HiGANplus/cGAN/model.py`
- Change the value of the `DiscModel.input_nc` config parameter in the file `~/HiGANplus/HiGAN+/configs/gan_iam.yml` on line `119` from `1` to `81`

Follow the steps mentioned in their [README file](https://github.com/ganji15/HiGANplus/blob/main/README.md) to train the model.
