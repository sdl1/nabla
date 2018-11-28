#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

def get_data(image_filename, labels_filename):
    # http://yann.lecun.com/exdb/mnist/
    with open(image_filename, "rb") as f:
        data = f.read()
    nsamples = int.from_bytes(data[4:8], byteorder="big")
    nrow = int.from_bytes(data[8:12], byteorder="big")
    ncol = int.from_bytes(data[12:16], byteorder="big")
    print("Found data nsamples={}, nrow={}, ncol={}".format(nsamples, nrow, ncol))
    images = np.zeros((nsamples, nrow*ncol))
    for i in range(nsamples):
        images[i, :] = np.frombuffer(data[16+i*nrow*ncol:16+(i+1)*nrow*ncol], dtype=np.dtype('>B'))

    with open(labels_filename, "rb") as f:
        data = f.read()
    nlabels = int.from_bytes(data[4:8], byteorder="big")
    print("Found {} labels".format(nlabels))
    labels = np.frombuffer(data[8:], dtype=np.dtype('>B'))

    return images, labels, nrow, ncol

def get_train_test_data():
    train_images_filename = "data/train-images-idx3-ubyte"
    train_labels_filename = "data/train-labels-idx1-ubyte"
    train_images, train_labels, nrow, ncol = get_data(train_images_filename, train_labels_filename)
    test_images_filename = "data/t10k-images-idx3-ubyte"
    test_labels_filename = "data/t10k-labels-idx1-ubyte"
    test_images, test_labels, _, _ = get_data(test_images_filename, test_labels_filename)
    return train_images, train_labels, test_images, test_labels, nrow, ncol
