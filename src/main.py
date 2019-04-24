# Program to implement a multilayer perceptron to classify hand-written digits from the MNIST database

# Package imports:
from mnist import MNIST
import numpy as np
import theano

# User imports:

# Main function:

# Configuration for 'python-mnist' package:
mndata = MNIST(path='./data', return_type='numpy')
mndata.test_img_fname = 't10k-images.idx3-ubyte'
mndata.test_lbl_fname = 't10k-labels.idx1-ubyte'
mndata.train_img_fname = 'train-images.idx3-ubyte'
mndata.train_lbl_fname = 'train-labels.idx1-ubyte'

# Data extraction:
train_images, train_labels = mndata.load_training()
test_images, test_labels = mndata.load_testing()
