import pickle
import numpy as np
from skimage import color

"""
Unpickles a batch of data. Pickle is a form of data compression.
Pickling method taken from https://www.cs.toronto.edu/~kriz/cifar.html, 
the dataset source.
"""


def unpickle(file):
    with open(file, 'rb') as fin:
        dict = pickle.load(fin, encoding='bytes')
    return dict


"""
Convert flattened image in RGB to CIE Lab. 
The image colorization algorithm by Zhang, Esola, and Efros 
(https://arxiv.org/pdf/1603.08511v5.pdf) uses images in CIE Lab 
because grayscale is trivial here ("L" in "Lab" stands for lightness,
which is the grayscale measure of the image)
"""


def toLab(img):
    square = img.T.reshape((32, 32, 3), order="F")  # convert into a square image
    square[:,:,0] = square[:,:,0].T
    square[:,:,1] = square[:,:,1].T
    square[:,:,2] = square[:,:,2].T
    return color.rgb2lab(square/255)  # convert from RGB to CIE Lab and return


"""
Processes one batch of data to be used with training or testing.
Output data is a dictionary with two entries:
- x: the input training data. Is the L* (lightness) channel for Lab color space
- y: the output result. Is the ground truth values for a* and b* channels for Lab color space

Parameters:
file - location of input batch - pickled data file
out_file - location to store processed data

"""


def processBatch(file, out_file):
    data = unpickle(file)
    new_dict = {"x": [], "y": []}
    for d in data[b'data']:
        lab = toLab(d)
        new_dict["x"].append(lab[:,:,0])
        new_dict["y"].append([lab[:,:,1], lab[:,:,2]])

    with open(out_file, 'wb') as fout:
        pickle.dump(new_dict, fout)


# Process all batches in the CIFAR data set
processBatch("data/cifar-10-batches-py/data_batch_1", "data/train_batch_1")
processBatch("data/cifar-10-batches-py/data_batch_2", "data/train_batch_2")
processBatch("data/cifar-10-batches-py/data_batch_3", "data/train_batch_3")
processBatch("data/cifar-10-batches-py/data_batch_4", "data/train_batch_4")
processBatch("data/cifar-10-batches-py/data_batch_5", "data/train_batch_5")
processBatch("data/cifar-10-batches-py/test_batch", "data/test_batch")
