# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import numpy as np
import zipfile
import PIL.Image
import json
import torch
import dnnlib
import pandas as pd  # <-- Add pandas for CSV reading

try:
    import pyspng
except ImportError:
    pyspng = None

#----------------------------------------------------------------------------

class Dataset(torch.utils.data.Dataset):
    def __init__(self,
        name,
        raw_shape,
        max_size    = None,
        use_labels  = False,
        xflip       = False,
        random_seed = 0,
        sampling_weights = None,    # <== Added argument for weights
    ):
        self._name = name
        self._raw_shape = list(raw_shape)
        self._use_labels = use_labels
        self._raw_labels = None
        self._label_shape = None

        self._sampling_weights = sampling_weights

        # Create raw index list
        self._raw_idx = np.arange(self._raw_shape[0], dtype=np.int64)

        # Apply sampling weights if provided
        if self._sampling_weights is not None:
            # Weighted sampling
            indices = np.arange(len(self._sampling_weights))
            weighted_indices = np.random.choice(
                indices, size=len(indices), replace=True, p=self._sampling_weights / self._sampling_weights.sum()
            )
            self._raw_idx = weighted_indices

        # Apply max_size
        if (max_size is not None) and (self._raw_idx.size > max_size):
            np.random.RandomState(random_seed).shuffle(self._raw_idx)
            self._raw_idx = np.sort(self._raw_idx[:max_size])

        # Apply xflip
        self._xflip = np.zeros(self._raw_idx.size, dtype=np.uint8)
        if xflip:
            self._raw_idx = np.tile(self._raw_idx, 2)
            self._xflip = np.concatenate([self._xflip, np.ones_like(self._xflip)])

    def _get_raw_labels(self):
        if self._raw_labels is None:
            self._raw_labels = self._load_raw_labels() if self._use_labels else None
            if self._raw_labels is None:
                self._raw_labels = np.zeros([self._raw_shape[0], 0], dtype=np.float32)
            assert isinstance(self._raw_labels, np.ndarray)
            assert self._raw_labels.shape[0] == self._raw_shape[0]
        return self._raw_labels

    def close(self):
        pass

    def _load_raw_image(self, raw_idx):
        raise NotImplementedError

    def _load_raw_labels(self):
        raise NotImplementedError

    def __getstate__(self):
        return dict(self.__dict__, _raw_labels=None)

    def __del__(self):
        try:
            self.close()
        except:
            pass

    def __len__(self):
        return self._raw_idx.size

    def __getitem__(self, idx):
        image = self._load_raw_image(self._raw_idx[idx])
        if self._xflip[idx]:
            image = image[:, :, ::-1]
        return image.copy(), self.get_label(idx)

    def get_label(self, idx):
        label = self._get_raw_labels()[self._raw_idx[idx]]
        return label.copy()

    @property
    def image_shape(self):
        return list(self._raw_shape[1:])

    @property
    def resolution(self):
        return self.image_shape[1]

    @property
    def label_shape(self):
        if self._label_shape is None:
            raw_labels = self._get_raw_labels()
            if raw_labels.dtype == np.int64:
                self._label_shape = [int(np.max(raw_labels)) + 1]
            else:
                self._label_shape = raw_labels.shape[1:]
        return list(self._label_shape)

#----------------------------------------------------------------------------

class ImageFolderDataset(Dataset):
    def __init__(self,
        path,
        resolution = None,
        **super_kwargs,
    ):
        self._path = path
        self._zipfile = None

        if os.path.isdir(self._path):
            self._type = 'dir'
            self._all_fnames = {os.path.relpath(os.path.join(root, fname), start=self._path)
                                for root, _dirs, files in os.walk(self._path) for fname in files}
        elif self._file_ext(self._path) == '.zip':
            self._type = 'zip'
            self._all_fnames = set(self._get_zipfile().namelist())
        else:
            raise IOError('Path must point to a directory or zip')

        PIL.Image.init()
        self._image_fnames = sorted(fname for fname in self._all_fnames
                                    if self._file_ext(fname) in PIL.Image.EXTENSION)
        if len(self._image_fnames) == 0:
            raise IOError('No image files found in the specified path')

        name = os.path.splitext(os.path.basename(self._path))[0]
        raw_shape = [len(self._image_fnames)] + list(self._load_raw_image(0).shape)

        # Load sampling weights from CSV if available
        sampling_weights = self._load_sampling_weights()

        if resolution is not None and (raw_shape[2] != resolution or raw_shape[3] != resolution):
            raise IOError('Images do not match specified resolution')

        super().__init__(name=name, raw_shape=raw_shape, sampling_weights=sampling_weights, **super_kwargs)

    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    def _get_zipfile(self):
        if self._zipfile is None:
            self._zipfile = zipfile.ZipFile(self._path)
        return self._zipfile

    def _open_file(self, fname):
        if self._type == 'dir':
            return open(os.path.join(self._path, fname), 'rb')
        elif self._type == 'zip':
            return self._get_zipfile().open(fname, 'r')
        return None

    def _load_sampling_weights(self):
        # Check if CSV is present
        csv_file = 'sampling_weights.csv'
        if csv_file in self._all_fnames:
            with self._open_file(csv_file) as f:
                df = pd.read_csv(f)
        elif os.path.exists(os.path.join(self._path, csv_file)):
            df = pd.read_csv(os.path.join(self._path, csv_file))
        else:
            return None

        # Ensure it has a 'weight' column and matches number of images
        weights = df['weight'].values
        if len(weights) != len(self._image_fnames):
            raise ValueError(f'Length of weights in sampling_weights.csv does not match number of images.')
        return weights

    def close(self):
        try:
            if self._zipfile is not None:
                self._zipfile.close()
        finally:
            self._zipfile = None

    def __getstate__(self):
        return dict(super().__getstate__(), _zipfile=None)

    def _load_raw_image(self, raw_idx):
        fname = self._image_fnames[raw_idx]
        with self._open_file(fname) as f:
            if pyspng is not None and self._file_ext(fname) == '.png':
                image = pyspng.load(f.read())
            else:
                image = np.array(PIL.Image.open(f))
        if image.ndim == 2:
            image = image[:, :, np.newaxis]
        image = image.transpose(2, 0, 1)
        return image

    def _load_raw_labels(self):
        fname = 'dataset.json'
        if fname not in self._all_fnames:
            return None
        with self._open_file(fname) as f:
            labels = json.load(f)['labels']
        labels = dict(labels)
        labels = [labels[fname.replace('\\', '/')] for fname in self._image_fnames]
        labels = np.array(labels)
        return labels.astype({1: np.int64, 2: np.float32}[labels.ndim])


#----------------------------------------------------------------------------
