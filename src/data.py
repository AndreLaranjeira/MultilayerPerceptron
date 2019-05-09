# Module to handle data conversion and processing.

# Package imports:
import numpy as np
import theano
import theano.tensor as Ttensor

# Public methods:

# Method to convert numpy data to theano shared variables.
def np_to_theano(data, cast_to_label=False, borrow=True):
    theano_data = theano.shared(np.asarray(data,
                                           dtype=theano.config.floatX),
                                           borrow=borrow)

    if(cast_to_label):
        return Ttensor.cast(theano_data, 'int32')

    else:
        return theano_data
