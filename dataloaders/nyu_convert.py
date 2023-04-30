import numpy as np
import scipy.io as sio
import h5py
import os

# Modify next 2 lines
data_path = "data/nyu"  # where to find nyu_depth_v2_labeled.mat and splits.mat
output_path = "data/nyu"  # where to put the resulting dataset

train_path = os.path.join(output_path, "train/official/")
val_path = os.path.join(output_path, "val/official/")

splilts = sio.loadmat(data_path + "/splits.mat")

train_idx = splilts["trainNdxs"]
val_idx = splilts["testNdxs"]

train_idx = np.array(train_idx)
val_idx = np.array(val_idx)

f = h5py.File(data_path + "/nyu_depth_v2_labeled.mat")
images = f["images"]
depths = f["depths"]
labels = f["labels"]

images = np.array(images)
depths = np.array(depths)
labels = np.array(labels)


if not os.path.isdir(train_path):
    os.makedirs(train_path)

if not os.path.isdir(val_path):
    os.makedirs(val_path)

for idx in range(len(train_idx)):
    f_idx = "{0:0>5}".format(int(train_idx[idx]))
    print("train:", f_idx)
    h5f = h5py.File(train_path + f_idx + ".h5", "w")

    h5f["rgb"] = np.transpose(images[train_idx[idx] - 1][0], (0, 2, 1))
    h5f["depth"] = np.transpose(depths[train_idx[idx] - 1][0], (1, 0))

    h5f.close()

for idx in range(len(val_idx)):
    f_idx = "{0:0>5}".format(int(val_idx[idx]))
    print("val:", f_idx)
    h5f = h5py.File(val_path + f_idx + ".h5", "w")

    h5f["rgb"] = np.transpose(images[val_idx[idx] - 1][0], (0, 2, 1))
    h5f["depth"] = np.transpose(depths[val_idx[idx] - 1][0], (1, 0))

    h5f.close()

print(train_idx[0])
print(images[train_idx[0] - 1][0].shape)
print(depths[train_idx[0] - 1][0].shape)
