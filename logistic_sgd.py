import numpy
import theano
import theano.tensor as T
import numpy as np
import psycopg2
from datetime import datetime

class LogisticRegression(object):
	def __init__(self,input,n_in,n_out,rng,zero_beta=False,dropout_rate=0.25,
		     amp_value = None):
		self.rng = rng
		self.W = theano.shared(
			value = numpy.zeros(
				(n_in,n_out),
				dtype=theano.config.floatX
				),
			name='W',
			borrow=True
		)
		self.b = theano.shared(
			value = numpy.zeros(
				(n_out,),
				dtype=theano.config.floatX
				),
			name='b',
			borrow=True
		)
		dropout_weight = theano.shared(
			np.asarray(1,dtype=theano.config.floatX))
		dropout_matrix = theano.shared(
			numpy.asarray(
				rng.binomial(1,1-dropout_rate,(n_in,n_out)),
				dtype=theano.config.floatX
				),
			name='d',
			borrow=True
		)
		self.n_in = n_in 
		self.n_out = n_out
		self.dropout_weight = theano.shared(
			np.asarray(1,dtype=theano.config.floatX))
		self.dropout_matrix = dropout_matrix
		self.dropout_rate = dropout_rate
		if amp_value == None:
			if not zero_beta:
				self.p_y_given_x = T.nnet.softmax(
					T.dot(input,self.W * self.dropout_matrix )\
						+ self.b)
			else:
				self.p_y_given_x = T.nnet.softmax(
					T.dot(input,self.W)*dropout_weight)
		else:
			if not zero_beta:
				self.p_y_given_x = T.nnet.softmax(
					T.dot(input,amp_value * self.W * \
						      self.dropout_matrix ) +self.b)
			else:
				self.p_y_given_x = T.nnet.softmax(
					T.dot(input,amp_value * self.W) * \
						dropout_weight)
		self.y_pred = T.argmax(self.p_y_given_x,axis=1)
		self.params = [self.W,self.b]
		self.input = input
		self.use_dropout=True
	def negative_log_likelihood(self,y):
		return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]),y])

        def turn_biases_off(self, reverse = False):
                """this is useful for production when you do not want to assume
                any inherent nature about the a priori likelihood of a particular
                class"""
                if not reverse:
                        self.b_copy = self.b.get_value()
                        self.b.set_value(np.float32(0. * self.b_copy))
                else:
                        self.b.set_value(self.b_copy)
                
	def errors(self,y):
		if y.ndim != self.y_pred.ndim:
			raise TypeError(
				'y should have the same shape as self.y_pred',
				('y', y.type,'y_pred',self.y_pred.type)
			)
		if y.dtype.startswith('int'):
			return T.mean(T.neq(self.y_pred,y))
		else:
			raise NotImplementedError()
	def reset_dropout_matrix(self, all_unit=False):
		if all_unit:
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

