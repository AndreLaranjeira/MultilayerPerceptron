# Module to implement a multilayer perceptron.

# Package imports:
import numpy as np
import theano

# Class to represent the output layer of a multilayer perceptron:
#   Parameters:
#       layer_input: TensorType with input variables for this layer.
#       n_inputs: Number of input variables.
#       n_outputs: Number of outputs (classes) desired.

class OutputLayer:

    def __init__(self, layer_input, n_inputs, n_outputs):

        # Initialize layer_input, weights and biases:
        self.layer_input = layer_input
        self.weights = theano.shared(value = np.zeros(
                                             (n_inputs, n_outputs),
                                             dtype=theano.config.floatX),
                                     name = 'weights',
                                     borrow = True)
        self.biases = theano.shared(value = np.zeros(
                                            (n_outputs,),
                                            dtype=theano.config.floatX),
                                    name = 'bias',
                                    borrow = True)

        # Record the parameters:
        self.params = [self.weights, self.biases]

        # Probability function:
        self.y_probabilities = T.nnet.softmax(T.dot(layer_input, self.weights)
                                              + self.biases)

        # Prediction function:
        self.predict = T.argmax(self.p_y_given_x, axis=1)

    # Function to compute the percentage of errors in an example mini-batch:
    def error_percentage(self, y):

        # First and foremost, error handling!
        # Handles different output and prediction sizes:
        if(y.ndim != self.y_pred.ndim):
            raise TypeError(
                'Label tensor is not the same size as predictions tensor!',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )

        # Check if labels are integers:
        if y.dtype.startswith('int'):
            return T.mean(T.neq(self.y_pred, y))

        # Handles labels that aren't integers:
        else:
            raise TypeError('Label tensors must be integers!')

    # Negative logarithm likehood function:
    def negative_log_likehood(self, y):
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])
