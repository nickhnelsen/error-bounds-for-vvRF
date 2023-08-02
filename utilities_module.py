"""
Modified from: https://github.com/zongyi-li/fourier_neural_operator/blob/master/utilities3.py
"""

import torch
import numpy as np
import scipy.io

#################################################
#
# Utilities (FOR BURGERS DATASET ONLY)
#
#################################################

def truncate_rfft2(mat, modes):
    """
    Truncates (K, K) array to (m, m) array, m<K, where m is even
    Args:
        modes (int): highest frequency to keep; m = ((modes + 1)//2)*4
    """
    modes = ((modes + 1)//2)*2
    if 2*modes > mat.shape[-2]:
        raise ValueError("modes is at most mat.shape[-2]//2")
    mat = mat[..., :modes + 1]
    return torch.cat((mat[...,:modes + 1,:], mat[...,-modes + 1:,:]), dim=-2)

def half_to_full(rfft_mat):
    """
    Takes Fourier mode weight matrix in half rfft format and contructs the full ``fft'' format matrix.
    Input
    ----
    rfft_mat: (..., 2*jmax, kmax + 1), matrix in rfft format with last dimension missing negative modes
        NOTE: 2*jmax must be even and kmax + 1 must be odd
        First index j ordering: j=0,1,...,jmax | -(jmax-1),-(jmax-2),...,-1
        Second index k ordering: k=0,1,...,kmax
    """    
    modes = torch.flip(rfft_mat[...,1:,1:-1].conj(), [-2,-1])
    modes = torch.cat((torch.flip(rfft_mat[..., 0:1, 1:-1].conj(), [-1]), modes), -2)
    return torch.cat((rfft_mat, modes), -1)

# Reference: https://discuss.pytorch.org/t/how-to-retrieve-the-sample-indices-of-a-mini-batch/7948/19
def dataset_with_indices(cls):
    """
    Modifies the given Dataset class to return a tuple data, target, index
    instead of just data, target.
    """

    def __getitem__(self, index):
        data, target = cls.__getitem__(self, index)
        return data, target, index

    return type(cls.__name__, (cls,), {'__getitem__': __getitem__,})

# reading data
class DataReader(object):
    def __init__(self, file_path, to_torch=True, to_cuda=False, to_float=True):
        super(DataReader, self).__init__()

        self.to_torch = to_torch
        self.to_cuda = to_cuda
        self.to_float = to_float

        self.file_path = file_path

        self.data = None
        self.old_mat = None
        self._load_file()

    def _load_file(self):
        try: # matlab .mat data file
            self.data = scipy.io.loadmat(self.file_path)
            self.old_mat = True
        except: # numpy .npy data file
            self.data = np.load(self.file_path)
            self.old_mat = False

    def load_file(self, file_path):
        self.file_path = file_path
        self._load_file()

    def read_field(self, field):
        if self.old_mat:
            x = self.data[field]

        elif not self.old_mat:
            if field == "a": # input coefficient functions
                x = self.data[...,0].T[...,:-1]
            elif field == "u": # output solution functions
                x = self.data[...,1].T[...,:-1]
            else:
                print("Error. Only enter `a' or `u' in the field argument")

        if self.to_float:
            x = x.astype(np.float32)

        if self.to_torch:
            x = torch.from_numpy(x)

        if self.to_cuda:
            x = x.cuda()
            
        return x

    def set_cuda(self, to_cuda):
        self.to_cuda = to_cuda

    def set_torch(self, to_torch):
        self.to_torch = to_torch

    def set_float(self, to_float):
        self.to_float = to_float


# normalization, pointwise gaussian
class UnitGaussianNormalizer(object):
    def __init__(self, x, eps=1e-6):
        super(UnitGaussianNormalizer, self).__init__()

        # x could be in shape of ntrain*n or ntrain*T*n or ntrain*n*T
        self.mean = torch.mean(x, 0)
        self.std = torch.std(x, 0)
        self.eps = eps

    def encode(self, x):
        x = (x - self.mean) / (self.std + self.eps)
        return x

    def decode(self, x, sample_idx=None):
        if sample_idx is None:
            std = self.std + self.eps # n
            mean = self.mean
        else:
            if len(self.mean.shape) == len(sample_idx[0].shape):
                std = self.std[sample_idx] + self.eps  # batch*n
                mean = self.mean[sample_idx]
            if len(self.mean.shape) > len(sample_idx[0].shape):
                std = self.std[:,sample_idx]+ self.eps # T*batch*n
                mean = self.mean[:,sample_idx]

        # x is in shape of batch*n or T*batch*n
        x = (x * std) + mean
        return x

    def cuda(self):
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()


# normalization, Gaussian
class GaussianNormalizer(object):
    def __init__(self, x, eps=1e-6):
        super(GaussianNormalizer, self).__init__()

        self.mean = torch.mean(x)
        self.std = torch.std(x)
        self.eps = eps

    def encode(self, x):
        x = (x - self.mean) / (self.std + self.eps)
        return x

    def decode(self, x, sample_idx=None):
        x = (x * (self.std + self.eps)) + self.mean
        return x

    def cuda(self):
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()


# normalization, scaling by range
class RangeNormalizer(object):
    def __init__(self, x, low=0.0, high=1.0):
        super(RangeNormalizer, self).__init__()
        mymin = torch.min(x, 0)[0].view(-1)
        mymax = torch.max(x, 0)[0].view(-1)

        self.a = (high - low)/(mymax - mymin)
        self.b = -self.a*mymax + high

    def encode(self, x):
        s = x.size()
        x = x.view(s[0], -1)
        x = self.a*x + self.b
        x = x.view(s)
        return x

    def decode(self, x):
        s = x.size()
        x = x.view(s[0], -1)
        x = (x - self.b)/self.a
        x = x.view(s)
        return x
