# Program to implement a multilayer perceptron to classify hand-written digits
# from the MNIST database

# Package imports:
from mnist import MNIST
import numpy as np
import sys
import theano
from os import listdir
import optparse

# User imports:
import data
import mlp
import plot

# Main function:

# Program arguments:
parser = optparse.OptionParser()
parser.add_option('-o', action='store', dest="filename", help="filename prefix to save plot images")
parser.add_option('--cross-entropy', action='store_true', dest="crossEntropy", help="sets cross entropy as cost function, default is mean squared error", default=False)
parser.add_option('--softmax', action='store_true', dest="softMax", help="sets softmax as activation function on output layer", default=False)
parser.add_option('--learning-rate', action='store', dest="learning_rate", type="float", help="sets training learning constant", default=0.01, metavar="NUM")
parser.add_option('--num-epoches', action='store', dest="n_epoches", type="int", help="sets number of epoches for training", default=10, metavar="NUM")
parser.add_option('--batch-size', action='store', dest="batch_size", type="int", help="sets batch size for training with minibatches", default=20, metavar="NUM")
parser.add_option('--L1-reg', action='store', dest="L1_reg", type="float", help="sets L1 regularization constant", default=0.0, metavar="NUM")
parser.add_option('--L2-reg', action='store', dest="L2_reg", type="float", help="sets L2 regularization constant", default=0.0001, metavar="NUM")
parser.add_option('--hidden-layers', action='store', dest="hiddenLayers", help="comma separated list with size for each hidden layer, example: --hidden-layers 100,100 (default). If --hidden-layers none, no hidden layers will be used", default="100,100", metavar="LIST")
options, args = parser.parse_args()

#if(len(sys.argv) != 3 and len(sys.argv) != 4):
#    print("[Error] Number of program arguments is wrong!")
#    print("Usage: python src/main.py learn_constant num_of_epochs [output_dir]")
#    sys.exit(1)

#learn_constant = float(sys.argv[1])
#num_of_epochs = int(sys.argv[2])
#output_dir = sys.argv[3] if len(sys.argv) == 4 else None

# Configuration for 'python-mnist' package:
print("Loading data... ", end='')
mndata = MNIST(path='./data', return_type='numpy')
mndata.test_img_fname = 't10k-images.idx3-ubyte'
mndata.test_lbl_fname = 't10k-labels.idx1-ubyte'
mndata.train_img_fname = 'train-images.idx3-ubyte'
mndata.train_lbl_fname = 'train-labels.idx1-ubyte'
print("(Done!)")

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

classifier, train_stats = mlp.train(
    input_data=train_images, 
    input_data_size=28*28, 
    input_label=train_labels, 
    n_output=10, 
    test_data=test_images, 
    test_label=test_labels, 
    hlayer_sizes=[] if options.hiddenLayers == "none" else [int(x) for x in options.hiddenLayers.split(',')],
    learning_rate=options.learning_rate,
    n_epochs=options.n_epoches,
    batch_size=options.batch_size,
    L1_reg=options.L1_reg,
    L2_reg=options.L2_reg,
    crossEntropy=options.crossEntropy,
    softMax=options.softMax
)

if options.filename is not None:
    plot_epochError = plot.data_plot([e['epoch_error'] for e in train_stats], 'Erro da época')
    plot_testError = plot.data_plot([e['test_error'] for e in train_stats], 'Erro do teste')
    plot.save_plot([plot_epochError, plot_testError], 'Erros', 'Nº da época', options.filename + '_error.png')

    plot_cost = plot.data_plot([e['cost'] for e in train_stats], 'Custo da época')
    plot.display_plot([plot_cost], 'Custo', 'Nº da época', options.filename + '_cost.png')

# Extra test images made by us
extra_examples_paths = [f for f in listdir('extra_examples') if f.split('.')[-1] == "raw"]
extra_images = data.np_to_theano(list(map(lambda path: list(open("extra_examples/" + path, "rb").read()), extra_examples_paths)))
extra_labels = [int(label.split('.')[0]) for label in extra_examples_paths]

prediction = list(classifier.predict(extra_images))

print("Result for examples made by us:")
print("Expected: ")
print(extra_labels)
print("Predicted: ")
print(prediction)

