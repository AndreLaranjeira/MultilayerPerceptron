# Module to implement a multilayer perceptron.

# Package imports:
import numpy as np
import theano
import theano.tensor as T

# Classes:

# Class to represent a complete multilayer perceptron (MLP):
#   Parameters:
#       input: TensorType with input variables for this MLP.
#       n_inputs: Number of input variables.
#       n_outputs: Number of outputs (classes) desired.
#       n_hlayers: Number of hidden layers.
#       hlayer_sizes: Array with the sizes of the hidden layers.
#           IMPORTANT: Must be the same length as n_hlayers!

class MultilayerPerceptron:

    def __init__(self, layer_input, n_inputs, n_outputs, n_hlayers, hlayer_sizes):

        # Initialize the input:
        self.input = layer_input

        # WARNING: This implementation is only temporary! Changes should be made
        # to it, to account for the HiddenLayer implementation!!!
        
        self.hidden_layers = []
        
        # Create n_hlayers of hidden layers
        for i in range(n_hlayers):
            # Append hiddenLayer
            self.hidden_layers.append(
                HiddenLayer(
                    rng=np.random.RandomState(1234), # Could be passed by argument
                    layer_input=self.hidden_layers[i-1].output if (i != 0) else layer_input,
                    n_in=hlayer_sizes[i-1] if (i != 0) else n_inputs,
                    n_out=hlayer_sizes[i],
                    activation=T.tanh
                )
            )
        
        if(n_hlayers == 0):
            output_layer_input = layer_input
            output_layer_n_inputs = n_inputs
        else:
            # Last hidden layer output
            output_layer_input = self.hidden_layers[-1].output
            # Last hidden layer size
            output_layer_n_inputs = hlayer_sizes[-1]
        
        
        # Initializing the output layer:
        self.output_layer = OutputLayer(layer_input = output_layer_input, 
                                        n_inputs = output_layer_n_inputs, 
                                        n_outputs = n_outputs)

        # Record the parameters:
        self.params = self.output_layer.params + list(sum(map(lambda x: x.params, self.hidden_layers),[]))

        # Cost metrics:
        ## L1 cost metric:
        self.L1 = abs(self.output_layer.W).sum() + sum(map(lambda x: x.W.sum(), self.hidden_layers))

        ## L2 squared cost metric:
        self.L2_sqr = (self.output_layer.W ** 2).sum() + sum(map(lambda x: (x.W ** 2).sum(), self.hidden_layers))

        # Classification metrics:
        ## MLP errors:
        self.error_percentage = self.output_layer.error_percentage

        ## MLP Negative logarithm likehood:
        self.negative_log_likehood = self.output_layer.negative_log_likehood
        
    def predict(self, layer_input):
        x = self.input
    
        predict_model = theano.function(
            inputs=[],
            outputs=self.output_layer.y_pred,
            givens={
                x: layer_input
            }
        )
        
        return predict_model()
        

# Class to represent the output layer of a multilayer perceptron:
#   Parameters:
#       layer_input: TensorType with input variables for this layer.
#       n_inputs: Number of input variables.
#       n_outputs: Number of outputs (classes) desired.

class OutputLayer:

    def __init__(self, layer_input, n_inputs, n_outputs):

        # Initialize layer_input, weights and biases:
        self.layer_input = layer_input
        self.W = theano.shared(value = np.zeros(
                                             (n_inputs, n_outputs),
                                             dtype=theano.config.floatX),
                                     name = 'W',
                                     borrow = True)
        self.b = theano.shared(value = np.zeros(
                                            (n_outputs,),
                                            dtype=theano.config.floatX),
                                    name = 'b',
                                    borrow = True)

        # Record the parameters:
        self.params = [self.W, self.b]

        # Probability function:
        self.p_y_given_x = T.nnet.softmax(T.dot(layer_input, self.W)
                                              + self.b)

        # Prediction function:
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

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
        
    # Squared error
    def squared_error(self, y):
        return T.sum((self.p_y_given_x-y) ** 2)
        
        
class HiddenLayer:
    
    def __init__(self, rng, layer_input, n_in, n_out, W=None, b=None, activation=T.tanh):
        
        # Set input object variable
        self.input = layer_input
        
        # Set weights using random number generator (rng) given
        if W is None:
            # Suggestion of Xavier
            W_values = np.asarray(
                rng.uniform(
                    low=-np.sqrt(6. / (n_in + n_out)),
                    high=np.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4
                
            # Set as theano variable
            W = theano.shared(value=W_values, name='W', borrow=True)
            
        # Set all initial biases to zero
        if b is None:
            b_values = np.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)
        
        self.W = W
        self.b = b
        
        lin_output = T.dot(layer_input, self.W) + self.b
        
        # Sets activation function over output if there's such function
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )
        
        self.params = [self.W, self.b]

def train(input_data, input_data_size, input_label, n_output, test_data, test_label, hlayer_sizes, learning_rate=0.01, L1_reg=0.00, L2_reg=0.0001, n_epochs=1000, batch_size=20):

    # Number of minibatches
    n_minibatches = input_data.get_value(borrow=True).shape[0] // batch_size
    
    #n_valid_batches = input_data.get_value(borrow=True).shape[0] // batch_size
    
    n_test_batches = test_data.get_value(borrow=True).shape[0] // batch_size
    
    # Minibatch index
    index = T.lscalar()
    
    # Input matrix
    x = T.matrix('x')
    
    # Label matrix
    y = T.ivector('y')
    
    # Instance of MLP
    classifier = MultilayerPerceptron(
        layer_input=x, 
        n_inputs=input_data_size, 
        n_outputs=n_output, 
        n_hlayers=len(hlayer_sizes), 
        hlayer_sizes=hlayer_sizes
    )
    
    # Cost definition
    cost = (
        classifier.negative_log_likehood(y)
        + L1_reg * classifier.L1
        + L2_reg * classifier.L2_sqr
    )
    
    # Compute gradient
    gparams = [T.grad(cost, param) for param in classifier.params]
    
    # Compute param update
    updates = [
        (param, param - learning_rate * gparam)
        for param, gparam in zip(classifier.params, gparams)
    ]
    
    # Train model definition
    train_model = theano.function(
        inputs=[index],
        outputs=[cost, classifier.error_percentage(y)],
        updates=updates,
        givens={
            x: input_data[index * batch_size : (index+1) * batch_size],
            y: input_label[index * batch_size : (index+1) * batch_size]
        }
    )
    
    # Validation model definition, validation set is the trainig set
#    validate_model = theano.function(
#        inputs=[index],
#        outputs=classifier.error_percentage(y),
#        givens={
#            x: input_data[index * batch_size : (index + 1) * batch_size],
#            y: input_label[index * batch_size : (index + 1) * batch_size]
#        }
#    )
    
    # Test model definition
    test_model = theano.function(
        inputs=[index],
        outputs=classifier.error_percentage(y),
        givens={
            x: test_data[index * batch_size : (index + 1) * batch_size],
            y: test_label[index * batch_size : (index + 1) * batch_size]
        }
    )
    
    print_freq = 1000
    test_score = 0.
    
    epoch = 0
    
    while (epoch < n_epochs):
        epoch = epoch + 1
        epoch_cost = 0
        epoch_error = 0
        
        for minibatch_index in range(n_minibatches):
            
            iter = epoch * n_minibatches + minibatch_index
            minibatch_cost, minibatch_error = train_model(minibatch_index)
            epoch_cost = epoch_cost + minibatch_cost
            epoch_error = epoch_error + minibatch_error * batch_size
            
#            if (iter % print_freq == 0):
#                print('    epoch %i/%i, minibatch %i/%i, minibatch cost %f' % (epoch, n_epochs, minibatch_index, n_minibatches, minibatch_cost))
    
        test_losses = [test_model(i) for i in range(n_test_batches)]
        test_score = np.mean(test_losses)
            
        print('epoch %i/%i, epoch cost %f, epoch error %f %%, test error %f %%' % (epoch, n_epochs, epoch_cost, epoch_error / (n_minibatches * batch_size) * 100, test_score * 100))
        
        if(epoch == n_epochs):
            train_more = input('Want more training? [y/n]\n')
            if train_more == 'y':
                extra_epochs = input('How many more epochs? ')
                n_epochs = n_epochs + int(extra_epochs)
    
    print('Optimization complete. test error %f %%' % (test_score * 100))
    
    return classifier

