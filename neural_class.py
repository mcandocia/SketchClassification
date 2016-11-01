import theano
from theano import tensor as T
from theano.tensor.nnet import conv2d
from theano.tensor.signal import downsample
from theano import function 
from logistic_sgd import LogisticRegression 

import timeit

import numpy as np
import numpy #I make too many typos

from mlp import HiddenLayer

import cPickle as pickle

import os




class NetConvPoolLayer(object):
	def __init__(self,rng,input,filter_shape,image_shape,
		     poolsize=(2,2),dropout_percent = 0.5,stride = None,
		     pool_method="max",amp_value = None, relu = False, 
		     batch_norm = False, precalculated_batchnorm_values = None, 
		     batchnorm_slide_percent = 0.):
		"""
		This is a convolutional layer that accepts 4D data and 
		convolutes the last 2 dimensions
		
		Args:
		    filter_shape: int array (#conv layers, #channels, width,height)
		    dropout_percent: float that randomly disables inputs, prevents
		        overfitting
		    
		    relu: Boolean that determines if output should be rectified
		batch_norm: Boolean that  determines if batch normalization 
		    should be  used
		precalculated_batchnorm_values: float array that will replace
		    per-batch calculated
		sliding_batchnorm_values: float, if nonzero, this will establish a
		    separate update to modify the means and standard deviation
		    after each batch, but not completely 
		"""

		try:
			assert image_shape[1] == filter_shape[1]
		except AssertionError:
			print 'Image shape is ' + str(image_shape[1]) 
			print 'Filter shape is ' + str(filter_shape[1])
			print "Poolsize is " + str(poolsize)
			raise AssertionError
		self.input = input
		self.precalculated_batchnorm_values = precalculated_batchnorm_values
		self.batchnorm_slide_percent = batchnorm_slide_percent
		#this is the size of the number of inputs to each
		#convolution hidden unit
		fan_in = np.prod(filter_shape[1:])
		#this is the number of output weights per channel divided by the
		#pooling size
		fan_out = (filter_shape[0] * np.prod(filter_shape[2:]))/\
		    np.prod(poolsize)
		W_bound = np.sqrt(6./(fan_in + fan_out))
		#initializes weights with appropriate values and shape
		self.W = theano.shared(
			np.asarray(
				rng.uniform(low=-W_bound,high=W_bound,
					    size=filter_shape),
				dtype=theano.config.floatX
				),
			borrow=True
		)
		dropout_matrix = theano.shared(
			np.asarray(
				rng.binomial(1,1-dropout_percent,filter_shape),
				dtype=theano.config.floatX
				),
			name='d',
			borrow=True
		)
		self.dropout_matrix = dropout_matrix
		#initializes biases to zero 
		#(important to avoid presumptions about content of image)
		b_values = numpy.zeros((filter_shape[0],),dtype=theano.config.floatX)
		#creates a shared variable instance (instead of a deep copy)
		self.b = theano.shared(value=b_values,borrow=True)
		self.dropout_weight = theano.shared(np.asarray(
				1.,dtype=theano.config.floatX))
		#convolve; dropout matrix should work, but keep an eye on it
		print 'conv image shape: ' + str(image_shape)
		self.params = [self.W,self.b]
		#track input (no longer redundant with batchnorm)
		self.raw_input = input
		if not batch_norm:
			self.input = input 
		else:
			print 'implementing batch normalization'
			rn_input = range(image_shape[1])
			#GAMMA is 0 because a constant of 1 is added to the transform
			#to avoid problems with the L2 norm
			self.GAMMA = theano.shared(np.float32([0. + rng.uniform(low=-0.001,high=0.001) for _ in rn_input]))
			self.BETA = theano.shared(np.float32([0 for _ in rn_input]))
			self.params += [self.GAMMA, self.BETA]
			if self.precalculated_batchnorm_values <> None:
				self.sd_input = self.precalculated_batchnorm_values[0]
				self.means = self.precalculated_batchnorm_values[1]
			elif self.batchnorm_slide_percent == 0:
				self.sd_input = T.sqrt(T.var(
						input,(0,2,3))+0.00001).dimshuffle('x',0,'x','x')
				self.means = T.mean(input,(0,2,3)).dimshuffle('x',0,'x','x')
			else:
				#set old values to initialized theano value
				self.sd_input_old = theano.shared(
					np.float32(
						np.ones(
							(1,image_shape[1],1,1)
							)
						),
					broadcastable=(True,False,True,True)
					)
				self.means_old = theano.shared(
					np.float32(
						np.zeros(
							(1,image_shape[1],1,1)
							)
						),
					broadcastable=(True,False,True,True)
					)
				sbsp = self.batchnorm_slide_percent
				self.sd_input = sbsp * self.sd_input_old + \
				    (1.-sbsp)*T.sqrt(T.var(input,(0,2,3))+0.00001).\
				    dimshuffle('x',0,'x','x')
				self.means = sbsp * self.means_old + \
				    (1-sbsp) * T.mean(input,(0,2,3)).\
				    dimshuffle('x',0,'x','x')
			self.input_normalized = (input - self.means)/self.sd_input
			self.input = self.input_normalized * (np.float32(1.) + \
				     self.GAMMA.dimshuffle('x',0,'x','x')) + \
			             self.BETA.dimshuffle('x',0,'x','x')
		if amp_value == None:
			conv_out = conv2d(
				input=self.input,
				filters=self.W*dropout_matrix,
				filter_shape=filter_shape,
				input_shape=image_shape#keyword changed from "image_shape"
			)
		else:
			conv_out = conv2d(
				input=self.input,
				filters=self.W*dropout_matrix * amp_value,
				filter_shape=filter_shape,
				input_shape=image_shape#keyword changed from "image_shape"
			)
		#pool (max pooling in this case)
		pooled_out = downsample.max_pool_2d(
			input = conv_out,
			ds = poolsize,
			ignore_border=True,
			st = stride,
			mode = pool_method
		)
		self.stride = stride
		self.pool_method = pool_method
                self.relu = relu
		x = T.tensor4('x',dtype = theano.config.floatX)
                linmax = function([x],T.maximum(x,0))
                if not self.relu:
                        self.output = T.tanh(pooled_out + 
                                             self.b.dimshuffle('x',0,'x','x'))
                else:
                        self.output = T.maximum(pooled_out + 
                                                self.b.dimshuffle('x',0,'x','x'),
                                                0)
		#store parameters
		#extra params to pass to other methods
		self.rng = rng 
		self.filter_shape = filter_shape
		self.dropout_percent = dropout_percent
		print 'conv filter shape: %s' % str(self.filter_shape)
	def reset_dropout_matrix(self,all_unit = False):
		if all_unit:
			self.dropout_weight.set_value(
				np.asarray(1. - self.dropout_percent,dtype=theano.config.floatX))
			self.dropout_matrix.set_value(
				np.asarray(
					np.ones(self.filter_shape)*(1-self.dropout_percent),
					dtype=theano.config.floatX
					))
		else:
			self.dropout_weight.set_value(np.asarray(1.,dtype=theano.config.floatX))
			self.dropout_matrix.set_value(
				np.asarray(
					self.rng.binomial(1,1-self.dropout_percent,self.filter_shape),
					dtype=theano.config.floatX
					))
	def get_unmodified_input(self):
		return self.raw_input.get_value()
