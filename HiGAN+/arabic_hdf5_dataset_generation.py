import itertools

import h5py
import numpy as np
import matplotlib.pyplot as plt
import os


def gen_h5file(all_imgs,
               all_texts,
               all_wids,
               save_name):
    """
    - all_imgs = ? vector, matrix with image data ?
    - all_texts = .txt with all text labels
    - all_wids =
    - save_name = name of .hdf5 file
    """
    img_seek_idxs, img_lens = [], []
    cur_seek_idx = 0
    for img in all_imgs:
        img_seek_idxs.append(cur_seek_idx)
        img_lens.append(img.shape[-1])
        print("img_lens", img.shape[-1])
        cur_seek_idx += img.shape[-1]

    lb_seek_idxs, lb_lens = [], []
    cur_seek_idx = 0
    for lb in all_texts:
        lb_seek_idxs.append(cur_seek_idx)
        lb_lens.append(len(lb))
        cur_seek_idx += len(lb)

    save_imgs = np.concatenate(all_imgs, axis=-1)
    save_texts = list(itertools.chain(*all_texts))
    save_lbs = [ord(ch) for ch in save_texts]
    save_path = os.path.join(save_name + '.hdf5')
    h5f = h5py.File(save_path, 'w')
    h5f.create_dataset('imgs',
                       data=save_imgs,
                       compression='gzip',
                       compression_opts=4,
                       dtype=np.uint8)
    h5f.create_dataset('lbs',
                       data=save_lbs,
                       dtype=np.int32)
    h5f.create_dataset('img_seek_idxs',
                       data=img_seek_idxs,
                       dtype=np.int64)
    h5f.create_dataset('img_lens',
                       data=img_lens,
                       dtype=np.int16)
    h5f.create_dataset('lb_seek_idxs',
                       data=lb_seek_idxs,
                       dtype=np.int64)
    h5f.create_dataset('lb_lens',
                       data=lb_lens,
                       dtype=np.int16)
    h5f.create_dataset('wids',
                       data=all_wids,
                       dtype=np.int16)
    h5f.close()
    print('save->', save_path)


def arabic_pictures_scaler():


    return


# # Open the HDF5 file
# # filename = 'data/iam/testset_words64_OrgSz.hdf5'
# filename = 'data/iam/trnvalset_words64_OrgSz.hdf5'
# f = h5py.File(filename, 'r')
#
# # Print the file information
# print('File name:', f.filename)
# print('File mode:', f.mode)
#
# # Access the root group
# root = f['/']
#
#
# # Print the structure of the file
# def print_structure(name, obj):
#     for key, val in obj.items():
#         if isinstance(val, h5py.Group):
#             print(name + key + '/')
#             print_structure(name + key + '/', val)
#         else:
#             print(name + key)
# print_structure('', root)
#
#
# # Print information about the datasets in the file
# for name in f:
#     print('Dataset name:', name)
#     dataset = f[name]
#     print('  Datatype:', dataset.dtype)
#     print('  Shape:', dataset.shape)
#     print(' ')
#
# # Close the file
# f.close()
#
#
# # Access the 'images' dataset
# img_lens = f['img_lens']
#
# # Retrieve the data as a numpy array
# images = np.array(img_lens)
#
# # Close the file
# f.close()
#
# # Do something with the data, such as display the first image
# plt.imshow(img_lens[0])
# plt.show()
