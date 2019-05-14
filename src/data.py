# Module to handle data conversion and processing.

# Package imports:
import numpy as np
import theano
import theano.tensor as Ttensor

# Public methods:

# Method to convert numpy data to theano shared variables.
def np_to_theano(data, cast_to_label=False, borrow=True):
    if(cast_to_label):
        theano_data = theano.shared(np.asmatrix([[1 if a == b else 0 for a in range(10)] for b in data],
                                               dtype=theano.config.floatX),
                                               borrow=borrow)
    else:
        theano_data = theano.shared(np.asarray(list(map(__normalize_data, data)),
                                               dtype=theano.config.floatX),
                                               borrow=borrow)

    return theano_data

# Private methods:
def __normalize_data(data):
    return (data - 127.5)/127.5
