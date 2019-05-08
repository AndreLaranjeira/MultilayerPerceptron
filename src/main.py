# Program to implement a multilayer perceptron to classify hand-written digits
# from the MNIST database

# Package imports:
from mnist import MNIST
import numpy as np
import sys
import theano

# User imports:
import mlp
import plot

# Main function:

# Program arguments:
if(len(sys.argv) != 3 and len(sys.argv) != 4):
    print("[Error] Number of program arguments is wrong!")
    print("Usage: python src/main.py learn_constant num_of_epochs [output_dir]")
    sys.exit(1)

learn_constant = float(sys.argv[1])
num_of_epochs = int(sys.argv[2])
output_dir = sys.argv[3] if len(sys.argv) == 4 else None

# Configuration for 'python-mnist' package:
mndata = MNIST(path='./data', return_type='numpy')
mndata.test_img_fname = 't10k-images.idx3-ubyte'
mndata.test_lbl_fname = 't10k-labels.idx1-ubyte'
mndata.train_img_fname = 'train-images.idx3-ubyte'
mndata.train_lbl_fname = 'train-labels.idx1-ubyte'

# Data extraction:
train_images, train_labels = mndata.load_training()
test_images, test_labels = mndata.load_testing()
