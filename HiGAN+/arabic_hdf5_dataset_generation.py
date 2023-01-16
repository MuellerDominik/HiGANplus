import itertools
import h5py
import numpy as np
import matplotlib.pyplot as plt
import os


def gen_h5file(all_imgs,
               all_texts,
               all_wids,
               save_name):

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
