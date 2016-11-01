import numpy as np
import os
import cPickle as pickle
from matplotlib import pyplot as plt 
import sys

border_size = 2
border = 2

def main(filename):
        with open(filename,'r') as f:
                objs = pickle.load(f)
        layers = objs['LAYER_VALUES']
        for layer in layers:
                if len(layer[0].shape) < 3:
                        continue
                viz_from_layer(layer[0])

def viz_layer(obj):
	shp = obj.shape 
        if len(shp) < 3:
                return 1
	row_amount = shp[1]
	image_height = shp[2]
	image_width = shp[3]
	col_amount = shp[0]
	fnums = row_amount * col_amount
	full_filter_image = np.zeros((row_amount * (border + image_height),
		col_amount * (image_width + border)))
	for filter_num in range(fnums):
		start_row = image_height * (filter_num/col_amount) +\
                            (filter_num/col_amount + 1)*border 
		end_row = start_row + image_height
		start_col = image_width * (filter_num % col_amount) +\
                            (filter_num % col_amount + 1)*border 
		end_col = start_col + image_width 
		full_filter_image[start_row:end_row,start_col:end_col] = \
		obj[filter_num % col_amount,filter_num // col_amount,:,:]
	return full_filter_image


def make_viz(viz):
	plt.imshow(viz)
	plt.axis('off')
	plt.set_cmap('spectral')
	plt.colorbar()
	plt.show()

def viz_from_layer(obj):
	make_viz(viz_layer(obj))
        
if __name__=='__main__':
        main(sys.argv[1])
