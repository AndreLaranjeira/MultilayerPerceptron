# Program to implement a multilayer perceptron to classify hand-written digits
# from the MNIST database

# Package imports:
from mnist import MNIST
import numpy as np
import sys
import theano

# User imports:
import data
import mlp
import plot

# Main function:

# Program arguments:
#if(len(sys.argv) != 3 and len(sys.argv) != 4):
#    print("[Error] Number of program arguments is wrong!")
#    print("Usage: python src/main.py learn_constant num_of_epochs [output_dir]")
#    sys.exit(1)

#learn_constant = float(sys.argv[1])
#num_of_epochs = int(sys.argv[2])
#output_dir = sys.argv[3] if len(sys.argv) == 4 else None

# Configuration for 'python-mnist' package:
mndata = MNIST(path='./data', return_type='numpy')
mndata.test_img_fname = 't10k-images.idx3-ubyte'
mndata.test_lbl_fname = 't10k-labels.idx1-ubyte'
mndata.train_img_fname = 'train-images.idx3-ubyte'
mndata.train_lbl_fname = 'train-labels.idx1-ubyte'

# Data extraction:
print("Extracting data... ", end='')

np_train_images, np_train_labels = mndata.load_training()
np_test_images, np_test_labels = mndata.load_testing()

print("(Done!)")

# Data formatting:
print("Formatting data... ", end='')

train_images = data.np_to_theano(np_train_images)
test_images = data.np_to_theano(np_test_images)
train_labels = data.np_to_theano(np_train_labels, cast_to_label=True)
test_labels = data.np_to_theano(np_test_labels, cast_to_label=True)

print("(Done!)")

classifier = mlp.train(
    input_data=train_images, 
    input_data_size=28*28, 
    input_label=train_labels, 
    n_output=10, 
    test_data=test_images, 
    test_label=test_labels, 
    hlayer_sizes=[20,20],
    learning_rate=0.001,
    n_epochs=10
)
    
