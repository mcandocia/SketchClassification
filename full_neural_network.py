import theano
import math
from theano import tensor as T
from theano import function
from theano.tensor.nnet import conv2d
from theano.tensor.signal import downsample

from logistic_sgd import LogisticRegression 
from neural_class import NetConvPoolLayer

import timeit
import time
from datetime import datetime

import numpy as np

import prepare_image_data

from mlp import HiddenLayer

import cPickle as pickle

import os
import sys
import neural_class
import psycopg2
import dbinfo

#VERY IMPORTANT - HANDLES ALL IMAGE FETCHING AND INITIAL TRANSFORMATIONS
import prepare_image_data

BATCH_NORMALIZATION_EPSILON = 0.00001

#parameters that are default are used to reconstruct the object when reloaded
#secondary parameters are values that can change throughout the course of training
default_params = ['batch_size','kernels','input_dimensions',
                  'convolution_dimensions','pool_sizes','stride_sizes',
                  'layer_pattern','relu_pattern','dropout_rate','rng_seed',
                  'base_learning_rate','learning_decay_per_epoch','l2_norm','name'
                  ,'param_index','address','n_epochs']

default_blacklist = ['momentum']

secondary_params = ['output_size','n_train_batches',
                    'n_valid_batches','n_test_batches','cat_labels',
                    'momentum_raw','learning_rate_raw','epoch',
                    'best_validation_loss','test_score','momentum_limit',
                    'constants_py']

#currently the learning rate is not any different than the other parameters
#hopefully low dimensionality shouldn't make a difference
batch_normalization_params = ['batch_norm_pattern','batch_norm_decay_per_epoch', 
                              'batchnorm_vals_filename', 
                              'batchnorm_sliding_percent']

#todo : debug batch normalization
batch_normalization_secondary_params = ['batch_norm_learning_rate_raw', 
                                        'uses_batch_normalization' ]

default_params += batch_normalization_params
secondary_params += batch_normalization_secondary_params

class neural_network(object):
    def __init__(self, batch_size, kernels, input_dimensions, 
                 convolution_dimensions, pool_sizes, stride_sizes, layer_pattern, 
                 relu_pattern,  dropout_rate,rng_seed=None, 
                 base_learning_rate = 0.05, momentum = 0.8,
                 learning_decay_per_epoch=0.91, l2_norm = 0,name="default",
                 param_index=0,address='',n_epochs = 200,
                 batch_normalization_pattern = None,batch_norm_learning_rate=0.1,
                 batch_norm_decay_per_epoch=0.95, batchnorm_vals_filename = None, 
                 batchnorm_slide_percent = 0., disable_fetchers=False):
        """
        batch_size - int - size of each batch
        kernels - int array - number of general units each layer (incl. input/output)
        input_dimensions - int array[2] -  dimensions of input
        convolution_dimensions - int array[2] array - dimensions of each convolution
        pool_sizes - int array[2] array - dimensions of pooling for each convolution
        stride_sizes - int array - length of strides for each convolutional layer (this overrides aspects of pooling behavior)
        layer_pattern - ['I','C',...,'C','F',...,'F','O'] - indicates pattern of layers
        relu_pattern - boolean array that describes if layers should be rectified
                       linear units (True) or tanh (False) or linear (None)
        dropout_rate - float - rate of dropout for network weights
        rng_seed - int - seed for random number generator; None defaults to random
        base_learning_rate - floatX - initial learning rate
        momentum - [floatX, floatX] - amount that learning rate carries over through
                   iterations, first item is initial value, last value is final val
        learning_decay_per_epoch - floatX - factor for increasing/decreasing 
                                   learning rate over epochs
        name - string that describes the beginning of the filenames of the network 
               pickle
        param_index - integer determined a priori to index the param configurations
                      and show it in the filename
        batchnorm_vals_filename - has to be constructed by separate file; 
                                  pre-defines mean and sd of each layer for a nn...
                                  might be preferred to use sliding instead, as 
        batchnorm_slide_percent - sort of like momentum, but for calculations of 
                                  batch-normalization means and standard deviations
        """
        with open('constants.py','r') as f:
            self.constants_py = f.read()
        self.batch_size = batch_size
        if not disable_fetchers:
            self.conn = psycopg2.connect("dbname=%s user=%s password=%s "\
                                         "host=%s port=%s" % 
                                         (dbinfo.dbname,dbinfo.user,
                                          dbinfo.password,dbinfo.host,
                                          dbinfo.port))
            self.cur = self.conn.cursor()
            self.fetcher = prepare_image_data.fetcher(self.batch_size)
            self.cat_labels = self.fetcher.valid_names
        else:
            #should only happen when loading a network (labels will be assigned
            #by the load function)
            self.cat_labels = None
        self.test_score = None
        #initialize arrays containing basic information and hyperparameters
        self.layers = []
        self.uses_batch_normalization = bool(batch_normalization_pattern)
        self.batch_norm_pattern = batch_normalization_pattern
        self.batchnorm_vals_filename = batchnorm_vals_filename
        self.batchnorm_slide_percent = batchnorm_slide_percent
        if not self.uses_batch_normalization:
            self.batch_norm_pattern = [False for _ in relu_pattern]
        self.address=address
        #replace future instances of self.kernel
        self.kernels = kernels
        self.input_dimensions = input_dimensions
        self.output_size = kernels[-1:][0]
        self.inputs = []
        self.x = x = T.ftensor4('x')
        self.y = y = T.ivector('y')
        self.rng = np.random.RandomState(rng_seed)
        self.name = name
        self.n_epochs = n_epochs
        self.shapes = [(input_dimensions[0],input_dimensions[1])]
        print "input shape: " + str(self.shapes)
        self.convolution_dimensions = convolution_dimensions
        self.rng_seed = rng_seed
        self.layer_pattern = layer_pattern
        self.current_batch_index = 0
        self.pool_sizes = pool_sizes
        self.stride_sizes = stride_sizes
        self.relu_pattern = relu_pattern
        #if the rate is a float, each layer has the same rate
        if type(dropout_rate) == type(1.1):
            dropout_rate = [dropout_rate for _ in layer_pattern]
        self.dropout_rate = dropout_rate
        
        self.learning_decay_per_epoch = learning_decay_per_epoch
        self.l2_norm = l2_norm

        #indexing information
        self.ratios = np.asarray([0.6,0.2,0.2])#currently unused
        self.index = index = T.lscalar()
        #temporarily hardcoded
	self.n_train_batches = 300
	self.n_valid_batches = 42
	self.n_test_batches = 32#testing is sorta pointless at this point
        self.momentum = theano.shared(np.float32(momentum[0]))
        self.momentum_limit = momentum[1]
        self.base_learning_rate = np.float32(base_learning_rate)
        self.learning_rate = theano.shared(np.float32(base_learning_rate))
        self.index = index = T.lscalar()
        self.momentum_raw = momentum[0]
        self.learning_rate_raw = self.learning_rate.get_value()
        if self.uses_batch_normalization:
            self.batch_norm_learning_rate_raw = batch_norm_learning_rate
            self.batch_norm_learning_rate = theano.shared(np.float32(
                    self.batch_norm_learning_rate_raw))
        self.epoch = 0
        #initialize basic file shapes
        #recent change: changed kernel_sizes to self.kernels
        self.training_x = theano.shared(
            np.zeros(
                shape = (batch_size,self.kernels[0],input_dimensions[0],
                         input_dimensions[1]),
                dtype = theano.config.floatX),
            borrow = True)
        self.input=self.x.reshape((self.batch_size,self.kernels[0],
                                   self.shapes[0][0],self.shapes[0][1]))
        #updated database-based retrieval
        self.training_y = theano.shared(
            np.zeros(
                shape=self.batch_size,
                dtype=np.int32),
            borrow=True)
        self.testing_x = theano.shared(
            np.zeros(
                shape=(
                    self.batch_size,kernels[0],
                    input_dimensions[0],
                    input_dimensions[1]
                    ),
                dtype = theano.config.floatX),
            borrow = True)
        self.testing_y =theano.shared(
            np.zeros(
                shape=self.batch_size,
                dtype = np.int32),
            borrow=True)
        self.validation_x = theano.shared(
            np.zeros(
                shape = (
                    self.batch_size,
                    kernels[0],
                    input_dimensions[0],
                    input_dimensions[1]
                    ),
                dtype = theano.config.floatX),
            borrow = True)
        self.validation_y = theano.shared(
            np.zeros(
                shape=self.batch_size,
                dtype=np.int32
                ),
            borrow=True)
        #load fixed mean and sd values if file exists
        if self.batchnorm_vals_filename <> None:
            self.batchnorm_fixed_values = pickle.load(self.batchnorm_vals_filename)
        else:
            self.batchnorm_fixed_values = [None for _ in range(len(layer_pattern))]
        ###begin creation of layers
        #I = "input";C = "Convolutional"; F = "Fully-Connected", O = "Output"
        for i, pattern in enumerate(layer_pattern):
            if pattern=="I":
                self.inputs.append(self.input)
                print 'inserted input'
            elif pattern=="C":
                
                self.layers.append(
                    NetConvPoolLayer(
                        self.rng,
                        input = self.inputs[i-1],
                        image_shape=(
                            batch_size,kernels[i-1],
                            self.shapes[i-1][0],
                            self.shapes[i-1][1]
                            ),
                        filter_shape=(
                            kernels[i],
                            kernels[i-1],
                            self.convolution_dimensions[i-1][0],
                            self.convolution_dimensions[i-1][1]),
                        poolsize = pool_sizes[i-1],
                        stride = stride_sizes[i-1],
                        dropout_percent = self.dropout_rate[i],
                        batch_norm = self.batch_norm_pattern[i],
                        relu=self.relu_pattern[i],
                        batchnorm_slide_percent = self.batchnorm_slide_percent,
                        precalculated_batchnorm_values = self.\
                            batchnorm_fixed_values[i-1])
                    )
                x_new = (
                    self.shapes[i-1][0] - self.convolution_dimensions[i-1][0] + \
                        1 - (pool_sizes[i-1][0] - stride_sizes[i-1][0]))/\
                        (stride_sizes[i-1][0]
                         )
                y_new = (
                    self.shapes[i-1][1] - self.convolution_dimensions[i-1][1] + 1 -\
                        (pool_sizes[i-1][1] - stride_sizes[i-1][1]))/\
                        (stride_sizes[i-1][1]
                         )
                self.inputs.append( self.layers[i-1].output ) 
                self.shapes.append((x_new,y_new))
                print "self.shapes: " + str( self.shapes ) 
                print 'added convolution layer'
            elif pattern=="F":
                if layer_pattern[i-1]=="C":
                    next_input = self.inputs[i-1].flatten(2)
                else:
                    next_input = self.inputs[i-1]
                if self.relu_pattern[i]:
                    hidden_activation = 'linmax'
                elif relu_pattern == None:
                    hidden_activation = None
                else:
                    hidden_activation = T.tanh
                self.layers.append(
                    HiddenLayer(
                        self.rng,
                        input=next_input,
                        n_in = kernels[i-1]*self.shapes[i-1][0]*self.shapes[i-1][1],
                        n_out = kernels[i],
                        activation = hidden_activation,
                        dropout_rate = self.dropout_rate[i]
                    )
                )
                self.inputs.append(self.layers[i-1].output)
                #the shape is only used to determine dimensions of the next layer
                self.shapes.append((1,1))#see if this fixes issue
                print 'added fully-connected hidden layer, shape=%s' %\
                    str(self.shapes[-1])
            else:
                if layer_pattern[i-1]=="C":
                    next_input = self.inputs[i-1].flatten(2)
                else:
                    next_input = self.inputs[i-1]
                self.layers.append(
                    LogisticRegression(
                        input = next_input,
                        n_in = kernels[i-1],
                        n_out = self.output_size,
                        rng = self.rng,
                        dropout_rate=self.dropout_rate[i])
                )
                last_index = i-1
                print 'added logistic layer'
        zero = np.float32(0.)
        self.L2_penalty = theano.shared(np.float32(l2_norm))
        self.params = params  = [param for layer in self.layers \
                                     for param in layer.params]
        self.cost = self.layers[last_index].negative_log_likelihood(self.y) +\
                    self.L2_penalty * (
                        T.sum([T.sum(self.layers[q].W * self.layers[q].W)\
                               for q in range(len(self.layers))]))
        #updating functions (incl. momentum)
        #update 1 (only used for derivation in update #4)
        self.old_updates = [theano.shared(zero * param_i.get_value())\
                                for param_i in params]
        self.current_delta = [theano.shared(np.float32(zero * param_i.get_value()))\
                                  for param_i in params]
        self.grads = T.grad(self.cost,params)
        #update 2
        self.current_change_update = [
            (current_delta_i, self.learning_rate * grad_i +\
                 self.momentum * old_updates_i)\
                for current_delta_i,grad_i, old_updates_i in\
                zip(self.current_delta,self.grads,self.old_updates)
            ]
        #update 3
        updates = [
            ( param_i,param_i - current_delta_i) for param_i, current_delta_i in\
                zip(params,self.current_delta)]
        #self.updates = []
        #update 4 (derived from update #1)
        momentum_updates = [(old_updates_i, current_delta_i)\
                                for old_updates_i, current_delta_i in\
                                zip(self.old_updates,self.current_delta)]
        #self.momentum_updates = []
        #now batch-normalization updates when needed
        batchnorm_sliding_updates = []
        for layer in self.layers:
            if not isinstance(layer,NetConvPoolLayer):
                continue
            if layer.batchnorm_slide_percent <> 0.:
                batchnorm_sliding_updates += [
                    (layer.sd_input_old, layer.sd_input),
                    (layer.means_old, layer.sd_input)
                    ]
        #combined updates
        self.all_updates = self.current_change_update + updates +\
            momentum_updates + batchnorm_sliding_updates
        #test model function
        self.test_model = theano.function(
            [],
                self.layers[last_index].errors(self.y),
                givens = {
                    x: self.testing_x,
                    y: self.testing_y
                }
        )
        #validation model function
        self.validate_model = theano.function(
            [],
                self.layers[last_index].errors(self.y),
                givens={
                    x:self.validation_x,
                    y:self.validation_y
                }
        )
        #training function
        self.train_model = theano.function(
            [],
                self.cost,
                updates = self.all_updates,
                givens={
                    x:self.training_x,
                    y:self.training_y
                }
        )
        self.patience = 20000
        self.patience_increase = 3
        self.improvement_threshold = 0.995
        self.validation_frequency = min(self.n_train_batches,self.patience//2)
        self.best_validation_loss = np.inf
        self.best_iter = 0
        #DEPRECATED 
        self.itermode = 'train'
        self.test_score = 0.
        self.start_time = timeit.default_timer()
        self.epoch = 0
        self.iter_i = 0 # renamed bc `iter` is reserved
        self.done_looping = False
        self.param_index = param_index
        #constant-defined stuff
        self.improvement_threshold=0.995
        self.validation_frequency = min(self.n_train_batches,self.patience//2)
        self.done_looping = False
        print 'initialized neural network object'
    def alter_momentum(self):
        self.momentum_raw = np.float32(self.momentum_limit * 0.07 + 
                                       self.momentum_raw * 0.93)
        self.momentum.set_value(self.momentum_raw)
    def alter_learning_rate(self, reduce_rate=True):
        current_rate = self.learning_rate_raw
        if reduce_rate:
            new_rate = np.float32(current_rate * self.learning_decay_per_epoch)
        else:
            new_rate = np.float32(current_rate / self.learning_decay_per_epoch)
        self.learning_rate_raw = new_rate
        self.learning_rate.set_value(new_rate)
    def run_iterations(self):
        print "running through iterations of %s" %  str(self)
        #run through these in case network just reloaded
        self.learning_rate.set_value(self.learning_rate_raw)
        print 'current learning rate: %s' % round(self.learning_rate_raw, 4)
        print 'current momentum: %s' % round(self.momentum_raw, 4)
        self.momentum.set_value(self.momentum_raw)
        #now begin actual algorithm
        start_time = timeit.default_timer()#check if lib loaded
        try:
            while (self.epoch < self.n_epochs) and not self.done_looping:
                loge = int(np.log(self.epoch+0.001)/np.log(10))
                lext = ''.join(['#' for _ in range(int(math.ceil(loge/2.)))])
                rext = ''.join(['#' for _ in range(int(math.floor(loge/2.)))])
                fext = lext + rext
                print "###########%s##########\n##%s####"\
                    "EPOCH %s "\
                    "###%s###\n#############%s########" % \
                    (fext,lext,self.epoch,rext,fext)
                print "training..."
                for train_i in range(self.validation_frequency):
                    self.fetch_training_samples()
                    self.reset_dropout_matrices()
                    cost_ij = self.train_model()
                    if train_i % 50 == 0:
                        sys.stdout.write('\n' + str(train_i) + ' | ' + 
                                         str(train_i + 
                                             (self.epoch) * 
                                             self.validation_frequency) + ' ~')
                    sys.stdout.write('.')
                    sys.stdout.flush()
                #going into validation
                self.reset_dropout_matrices(True)
                print "\nvalidating..."
                validation_losses = []
                for valid_i in range(self.n_valid_batches):
                    if valid_i % 100 == 0:
                        print valid_i
                    self.fetch_validation_samples()
                    validation_losses.append(self.validate_model())
                    self.predict_and_store(self.validation_x.get_value(), 
                                           self.validation_y.get_value())
                this_validation_loss = np.mean(validation_losses)
                print "Validation loss: %s" % this_validation_loss
                print datetime.now().strftime('%x %X')
                if this_validation_loss < self.best_validation_loss * \
                        self.improvement_threshold:
                    self.patience = max(self.patience, 
                                        (self.epoch + 1) * \
                                            self.validation_frequency * \
                                            self.patience_increase)
                    self.best_validation_loss = this_validation_loss
                    ##increases learning rate if not later in algorithm
                    if self.epoch < 28:
                        self.alter_learning_rate(reduce_rate=False)
                    else:
                        self.alter_learning_rate(reduce_rate=True)
                    print "testing..."
                    test_losses = []
                    for test_i in range(self.n_test_batches):
                        self.fetch_testing_samples()
                        if test_i % 100 == 0:
                            print test_i
                        test_losses.append(self.test_model())
                    test_score = np.mean(test_losses)
                    self.test_score = test_score
                    print "Test loss: %s" % test_score
                    self.save_network(mode='b')
                elif self.patience < self.epoch * self.validation_frequency:
                    self.done_looping = True
                else:
                    ##reduce learning rate
                    self.alter_learning_rate()
                self.epoch += 1
                self.alter_momentum()
                print "epoch ended\n learning rate: %s\nmomentum: %s" %\
                    (round(self.learning_rate_raw,4),
                     round(self.momentum_raw,4))
                self.save_network(epoch=True)





        except KeyboardInterrupt:
            print "Saving file after interruption"
            self.save_network(mode='i')
            print datetime.now().strftime('%x %X')
        print 'final loop. saving network...'
        self.save_network(mode='f')
        print datetime.now().strftime('%x %X')
                        
    def fetch_training_samples(self):
        samples = self.fetcher.fetch_training()
        self.training_x.set_value(samples[0])
        self.training_y.set_value(samples[1])
    def fetch_validation_samples(self):
        samples = self.fetcher.fetch_validation()
        self.validation_x.set_value(samples[0])
        self.validation_y.set_value(samples[1])
    def fetch_testing_samples(self):
        samples = self.fetcher.fetch_testing()
        self.testing_x.set_value(samples[0])
        self.testing_y.set_value(samples[1])
    def reset_dropout_matrices(self,weighted=False):
        for layer in self.layers:
            layer.reset_dropout_matrix(weighted)
    def __str__(self):
        return "neural network"
    def train_cost(self,x,y):
        return self.train_model(self.minibatch_index)
    def test_data(self,x,y):
        pass
    def save_network(self,mode=None, epoch=False):
        nd = dict()
        paramdict = dict()
        extraparams = []
        secondary_extraparams = []
        layer_params_batch = []
        #double-definition unnecessary
        if self.uses_batch_normalization:
            extraparams = batch_normalization_params
            secondary_extraparams = batch_normalization_secondary_params
        for param in default_params + extraparams:
            evalstring = 'paramdict["%s"] = self.%s' % (param , param)
            if not hasattr(self,param):
                continue
            #print evalstring
            exec(evalstring)
        for param in secondary_params + secondary_extraparams:
            evalstring = 'paramdict["%s"] = self.%s' % (param, param)
            if not hasattr(self, param):
                continue
            exec(evalstring)
        layer_params = []
        for i, layer in enumerate(self.layers):
            layer_params.append([layer.W.get_value(),layer.b.get_value()])
            if hasattr(layer,'GAMMA'):
                layer_params_batch.append(
                    [layer.GAMMA.get_value(),layer.BETA.get_value()]
                    )
            else:
                layer_params_batch.append(None)
        paramdict['LAYER_VALUES'] = layer_params
        paramdict["BATCH_LAYER_VALUES"] = layer_params_batch
        #print paramdict.keys()
        with open(self.construct_filename(mode, epoch),'w') as f:
            pickle.dump(paramdict,f)
            print "Saved parameters in file as \"%s\"" % \
                self.construct_filename(mode, epoch)
        #gives time for subsequent KeyboardInterrupt without interrupting dump
        time.sleep(3)
    """loads network from scratch (not an instance method)"""
    def construct_filename(self,mode='', epoch=None):
        """mode is used to determine at what point it is saved
        (e.g., best network, end of routine, after cancelling)"""
        subdirectory = ''
        if epoch:
            mode = 'epoch_%d_' % self.epoch
            subdirectory = 'neural_pickle_history/'
        return self.address + subdirectory + self.name + '_' + mode + \
            'network_' + str(self.param_index) + '.pickle'

    def predict(self,data,return_type=['class','number','probs'], 
                zero_biases=False):
        self.reset_dropout_matrices(True)
        if zero_biases:
            self.layers[-1].turn_biases_off()
        output = self.layers[-1].p_y_given_x.eval({self.x:data})
        res = dict()
        if 'probs' in return_type:
            res['probs'] = output
        nums = np.argmax(output,1)
        if 'number' in return_type:
            res['number'] = nums
        if 'class' in return_type:
            res['class'] = [self.cat_labels[x] for x in nums]
        return res

    def predict_and_store(self, data, y):
        """used to store predictions of data while validating"""
        #actual numeric class
        #predicted numeric class
        #timestamp
        #actual labeled class
        #predicted labeled class
        results = self.predict(data, return_type=['number','class'])
        numbers = results['number']
        classes = results['class']
        timestamp = datetime.now()
        for i, yval in enumerate(y):
            self.cur.execute("INSERT INTO errors VALUES(%s,%s,%s,%s,%s,%s,%s)",
                             (int(yval), int(numbers[i]), timestamp,
                             self.cat_labels[int(yval)],
                             classes[i], self.epoch, 
                             '_'.join(self.name, str(self.param_index))))
        self.conn.commit()

#batch size may need to be changed depending on purposes of network
def load_network_isolate(filename,modified_batch_size=None, disable_fetchers=False):
    #load pickle
    print "Loading %s" % (filename)
    with open(filename,'r') as f:
        paramdict = pickle.load(f)
    try:
        paramdict['batch_normalization_pattern'] = \
            paramdict.pop('batch_norm_pattern')
    except KeyError:
        print 'no batch_norm_pattern to pop'
    paramdict['disable_fetchers'] = disable_fetchers
    if modified_batch_size:
        paramdict['batch_size'] = modified_batch_size
    #using kwargs improved readability; should test
    newkwargs = {key:entry for key, entry in paramdict.iteritems() if key not in (
            ['LAYER_VALUES','BATCH_LAYER_VALUES'] + 
            default_blacklist + 
            secondary_params + 
            batch_normalization_secondary_params
            )
                 }
    print newkwargs
    network = neural_network(**newkwargs)
    '''
    estring = "network = neural_network(" + ','.join([" %s = %s" % (key,repr(entry)) for key, entry in paramdict.iteritems() if key not in [ "LAYER_VALUES","BATCH_LAYER_VALUES",'batch_norm_learning_rate' ]  and key not in default_blacklist and key not in secondary_params + batch_normalization_secondary_params]) + ')'
    #print estring
    #exec(estring)
    '''
    for values, layer in zip(paramdict['LAYER_VALUES'],network.layers):
        layer.b.set_value(values[1])
        layer.W.set_value(values[0])
    if 'BATCH_LAYER_VALUES' in paramdict:
        for values, layer in zip(paramdict['BATCH_LAYER_VALUES'],network.layers):
            if values <> None:
                layer.GAMMA.set_value(values[0])
                layer.BETA.set_value(values[1])
    for param, value in paramdict.iteritems():
        if param not in secondary_params + batch_normalization_secondary_params:
            #print '%s not in secondary_params' % param
            continue
        try:
            setattr(network,param,value)
            print 'set %s=%s' % (param, value)
            #exec("network.%s = %s" % (param, value))
        except:
            print param
            raise
    try:
        network.learning_rate_raw = paramdict['learning_rate_raw']
    except:
        print paramdict.keys()
        print 'issue with learning rate'
    return network
