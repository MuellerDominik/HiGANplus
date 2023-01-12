import h5py
import cv2

# Open the HDF5 file
with h5py.File('data/iam/testset_words64_OrgSz.hdf5', 'r') as f:
    # Access the data in the file
    img_lens = f['img_lens'][()]
    img_seek_idxs = f['img_seek_idxs'][()]
    imgs = f['imgs'][()]
    lb_lens = f['lb_lens'][()]
    lb_seek_idxs = f['lb_seek_idxs'][()]
    lbs = f['lbs'][()]
    wids = f['wids'][()]

    # cv2.imwrite(imgs[])

    i = 0
    print("img_lens:", img_lens[i])
    print("img_seek_idxs:", img_seek_idxs[i])
    print("imgs:", imgs[i])
    print("lb_lens:", lb_lens[i])
    print("lb_seek_idxs:", lb_seek_idxs[i])
    print("lbs:", lbs[i])
    print("wids:", wids[i])



# Note:
# File
# name: data / iam / trnvalset_words64_OrgSz.hdf5
# File
# mode: r

# Dataset
# name: img_lens
# Datatype: int16
# Shape: (52231,)
# Chunks: None
#
# Dataset
# name: img_seek_idxs
# Datatype: int64
# Shape: (52231,)
# Chunks: None
#
# Dataset
# name: imgs
# Datatype: uint8
# Shape: (64, 8139273)
# Chunks: (1, 127177)
#
# Dataset
# name: lb_lens
# Datatype: int16
# Shape: (52231,)
# Chunks: None
#
# Dataset
# name: lb_seek_idxs
# Datatype: int64
# Shape: (52231,)
# Chunks: None
#
# Dataset
# name: lbs
# Datatype: int32
# Shape: (248557,)
# Chunks: None
#
# Dataset
# name: wids
# Datatype: int16
# Shape: (52231,)
# Chunks: None
