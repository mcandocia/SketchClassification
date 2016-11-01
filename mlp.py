import os
import sys
import timeit
import numpy
import theano
import theano.tensor as T
import numpy as np

class HiddenLayer(object):
	def __init__(self,rng,input,n_in,n_out,W=None,
		b=None,activation=T.tanh,dropout_rate = 0.4,amp_value=None):
		if W is None:
			w_range = numpy.sqrt(6./(n_in + n_out))
			W_values = numpy.asarray(
				rng.uniform(
					low = -w_range,
					high = w_range,
					size = (n_in,n_out)
					),
				dtype = theano.config.floatX
			)
			if activation == theano.tensor.nnet.sigmoid:
				W_values *= 4
			W = theano.shared(value = W_values,name='W',borrow=True)
		if b is None:
			b_values = numpy.zeros((n_out,),dtype = theano.config.floatX)
			b = theano.shared(
				value=b_values,
				name='b' + str(n_out),
				borrow=True
				)
		self.W = W
		self.b = b 
		dropout_matrix = theano.shared(
			numpy.asarray(
				rng.binomial(1,1-dropout_rate,(n_in,n_out)),
				dtype=theano.config.floatX
				),
			name='d',
			borrow=True
		)
		self.dropout_weight = theano.shared(
			np.asarray(1.,dtype=theano.config.floatX)
			)
		self.dropout_matrix = dropout_matrix
		if amp_value==None:
			lin_output = T.dot(input,self.W * dropout_matrix) + self.b 
		else:
			lin_output = T.dot(input,amp_value * \
                                           self.W * dropout_matrix) + self.b
                if activation is None:
                        self.output = lin_output
                elif activation=='linmax':
                        self.output = T.maximum(lin_output, 0)
                else:
                        self.output = activation(lin_output)
		self.params = [self.W,self.b]
		self.dropout_rate = dropout_rate
		self.rng = rng
		self.n_in = n_in
		self.n_out = n_out
	def reset_dropout_matrix(self,all_unit = False):
		if all_unit:
			self.dropout_weight.set_value(np.asarray(
				1.-self.dropout_rate,dtype=theano.config.floatX))
			self.dropout_matrix.set_value(
				numpy.asarray(
					np.ones((self.n_in,self.n_out)) * \
						(1.-self.dropout_rate),
					dtype=theano.config.floatX
					)
			)
		else:
			self.dropout_weight.set_value(
				np.asarray(1.,dtype=theano.config.floatX)
				)
			self.dropout_matrix.set_value(
				numpy.asarray(
					self.rng.binomial(1,1-self.dropout_rate,
							  (self.n_in,self.n_out)),
					dtype=theano.config.floatX
					)
			)
