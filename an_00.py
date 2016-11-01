import full_neural_network as fnn
import math
import prepare_image_data
import numpy as np
import sys
#200 is fine

#fixed normalization issue

#trying same parameterization as current (better) network on server
#actual parameters
batch_size_c = 41
kernels_c = [4,44,48,60,80,64,57]
input_dimensions_c = [200,200]
convolution_dimensions_c = [(15,15),(7,7),(5,5)]
pool_sizes_c = [(6,6),(2,2),(2,2)]
stride_sizes_c = [(4,4),(2,2),(2,2)]
layer_pattern_c = ['I','C','C','C','F','F','O']
relu_pattern_c = [False,False,False,False,False,False,False]
dropout_rate_c = [0.0,0.5,0.5,0.45,0.4,0.4,0.25]
rng_seed_c = 723
base_learning_rate_c = 0.20
momentum_c = [0.5,0.9]#new momentum scales
learning_decay_per_epoch_c = 0.92
name_c = 'asketch_model'
param_index_c = 0
address_c = '/home/max/workspace/Sketch2/'
l2_norm_c = 0.0003

#CHANGELOG: 
LOAD = False

#constant warping is greatly reduced for this one
#NOTE: divided L2 norm by 3 to allow better generalization
def main(LOAD=False):
    if not LOAD:
        network = fnn.neural_network(batch_size = batch_size_c,
kernels = kernels_c, input_dimensions = input_dimensions_c,
convolution_dimensions = convolution_dimensions_c,pool_sizes = pool_sizes_c,
stride_sizes = stride_sizes_c, layer_pattern = layer_pattern_c,
relu_pattern = relu_pattern_c,dropout_rate = dropout_rate_c,
rng_seed = rng_seed_c, base_learning_rate = base_learning_rate_c,
momentum = momentum_c, learning_decay_per_epoch = learning_decay_per_epoch_c,
name=name_c,param_index = param_index_c,address = address_c,l2_norm = l2_norm_c)
        print 'created network'
    else:
        network_filename = 'sketch_model_inetwork_%d.pickle' % param_index_c
        network = fnn.load_network_isolate(network_filename, batch_size_c)
        print 'loaded network'
    network.run_iterations()
    return 0

if __name__=='__main__':
    if len(sys.argv) > 1:
        arg = sys.argv[1]
    else:
        arg = None
    main(arg in ["True",'true','T','t','TRUE'])
