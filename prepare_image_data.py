import pickle
import os
import numpy as np
from PIL import Image, ImageDraw
import re
from random import shuffle, choice
import theano
import math
import datetime
import copy
import psycopg2
import dbinfo
from distort_wheel import distortion_wheel
import cv2
from export_filenames_to_postgres import valid_names as VALID_NAMES
from transform_images import new_directory as TRANSFORMED_DIRECTORY
from google_collections.transform_collected import TRANSFORMED_DIRECTORY as GOOGLE_TRANSFORMED_DIRECTORY
xdim = 240
ydim = 240

cropx = 200
cropy = 200

WHEEL_MODIFICATIONS = 1

class fetcher:
	def __init__(self,batch_size,generate_distorter = True):
		self.train_conn = psycopg2.connect("dbname=%s user=%s password=%s host=%s port=%s" % (dbinfo.dbname,dbinfo.user,dbinfo.password,dbinfo.host,dbinfo.port))
		self.train_cursor = self.train_conn.cursor()
		self.test_conn = psycopg2.connect("dbname=%s user=%s password=%s host=%s port=%s" % (dbinfo.dbname,dbinfo.user,dbinfo.password,dbinfo.host,dbinfo.port))
		self.test_cursor = self.test_conn.cursor()
		self.valid_conn = psycopg2.connect("dbname=%s user=%s password=%s host=%s port=%s" % (dbinfo.dbname,dbinfo.user,dbinfo.password,dbinfo.host,dbinfo.port))
		self.valid_cursor = self.valid_conn.cursor()
		self.valid_names = VALID_NAMES
		self.regex = '__.*|'.join(self.valid_names)+'__.*'
		#set up queries for fetching information
		self.queries =[ "SELECT * FROM images_transformed WHERE role=%s ORDER BY random();" % i for i in [1,2,3]]
		#remove this when going back to training mode
		self.queries[0] = "SELECT * FROM images_transformed WHERE role<4"\
                                  "ORDER BY random()"
                #role=4 is image data that 
                self.queries[1] = self.queries[2] = """SELECT * FROM images_transformed WHERE role < 4;"""
		#end remove
		self.train_cursor.execute(self.queries[0])
		self.test_cursor.execute(self.queries[1])
		self.valid_cursor.execute(self.queries[2])
		#this modifies the images
		self.batch_size = batch_size
		#this will be disabled when the fetcher is run in prediction mode
		#or if distortions have been disabled in training (not currently
		#supported)
		if generate_distorter:
			self.distortion_wheel = distortion_wheel(xdim,ydim)
        def fetch_general(self, modify_wheel=True, divide=False, use_class=True):
                if modify_wheel:
			self.distortion_wheel.rotate_values(WHEEL_MODIFICATIONS)
		res = self.train_cursor.fetchmany(self.batch_size)
		num_results = len(res)
		if num_results < self.batch_size:
			self.train_cursor.execute(self.queries[1])
			res += self.train_cursor.fetchmany(self.batch_size -
                                                           num_results)
		return self.process_db_results(res,self.distortion_wheel,
                                               divide=divide)
	def fetch_training(self,modify_wheel=True):
		if modify_wheel:
			self.distortion_wheel.rotate_values(WHEEL_MODIFICATIONS)
		res = self.train_cursor.fetchmany(self.batch_size)
		num_results = len(res)
		if num_results < self.batch_size:
			self.train_cursor.execute(self.queries[1])
			res += self.train_cursor.fetchmany(self.batch_size - num_results)
		return self.process_db_results(res,self.distortion_wheel)
	def fetch_testing(self):
		res = self.test_cursor.fetchmany(self.batch_size)
		num_results = len(res)
		if num_results < self.batch_size:
			self.test_cursor.execute(self.queries[1])
			res += self.test_cursor.fetchmany(self.batch_size - num_results)
		return self.process_db_results(res)
	def fetch_validation(self):
		res = self.valid_cursor.fetchmany(self.batch_size)
		num_results = len(res)
		if num_results < self.batch_size:
			self.valid_cursor.execute(self.queries[1])
			res += self.valid_cursor.fetchmany(self.batch_size - num_results)
		return self.process_db_results(res)
	def process_db_results(self, res, distorter = None, divide=True):
		xlist = []
		ylist = []
		for r in res:
                        if r[2] < 4:
                                src_img = cv2.imread(TRANSFORMED_DIRECTORY + r[1],
                                                     cv2.IMREAD_UNCHANGED)
                        else:
                                src_img = cv2.imread(GOOGLE_TRANSFORMED_DIRECTORY\
                                                     + r[1], cv2.IMREAD_UNCHANGED)
			obj_class = r[0]
                        #note: distorted is distortion_wheel class
			if distorter:
				src_img = distorter.process_image(src_img,
                                                                  obj_class,
                                                                  smart=True,
                                                                  imname=r[1])
			src_img = crop_image(src_img,cropx,cropy)
			#convert RGB to LAB
                        #cannot do this with RGBA
			#src_img = cv2.cvtColor(src_img, cv2.COLOR_RGB2LAB)
			xlist.append(src_img)
			ylist.append(obj_class)
		return (np.float32(np.asarray(xlist)/255.),
                        np.int32(np.asarray(ylist)))	

def process_images(res,distorter = None,divide=True):
	xlist = []
	ylist = []
	for r in res:
		src_img = cv2.imread(r)
		if distorter:
			src_img = distorter.process_image(src_img)
		src_img = crop_image(src_img,cropx,cropy)
		if divide:
			src_img = np.float32(src_img/255.)
		xlist.append(src_img)
	return np.asarray(xlist)


def crop_image(img,cropx,cropy):
	shp = img.shape[0:2]
	xdiff = shp[0] - cropx
	ydiff = shp[1] - cropy
	img = img[xdiff//2:shp[0]-xdiff//2,ydiff//2:shp[1] - ydiff//2,:]
	return img


def random_rgb():
	return [np.random.randint(255) for _ in range(3)] 

def stroke_geom_values(img_size,imp_shape):
	center = [np.random.randint(size) for size in img_size]
	angle = np.random.normal(0,math.pi)
	length = np.maximum(1,np.random.normal(imp_shape['length_mean'],imp_shape['length_sd']))
	center2 = [int(np.round(center[0] + np.cos(angle)*length)),int(np.round(center[1] + np.sin(angle)*length))]
	radius = int(np.minimum(imp_shape['max_radius'],np.maximum(1,np.random.normal(imp_shape['radius_mean'],imp_shape['radius_sd']))))
	return {'center':center, 'center2': center2, 'radius':radius}


#add strokes to image as imperfections to help generalize the algorithm
#use 
def add_imperfections(image, max_num_imps = 5, 
	imp_shape = {'length_mean':12,'length_sd':10,'radius_mean':6,'radius_sd' : 4,'max_radius' : 18},
	prob_alpha=0.3,prob_palette=0.8):
	"""
	image - the RGBA numpy array
	max_num_imps - the range of number of imperfections to put on an image
	imp_shape - dictionary containing the following:
		length_mean - the average length of an imperfection stroke
		length_sd - the standard deviation of the length of a stroke; must be >= 1 (1 is a circle)
		radius_mean - the mean radius of a stroke
		radius_sd - the standard deviation of stroke radius; radius must be >=1
		max_radius - maximum allowed radius
	prob_alpha - probability that alpha will be used instead of a random color 
	prob_palette - probability that a chosen color comes from the palette of an image
	"""
	number_strokes = np.random.randint(max_num_imps)
	if not number_strokes:
		return 1
	size = image.size
	draw = ImageDraw.Draw(image)
	palette = image.getcolors(20000)
	palette = [x[1] for x in palette if x[1][3] > 250]
	if len(palette) == 0:
		palette = ((0,0,0,255))
	for i in range(number_strokes):
		geom_vals = stroke_geom_values(size,imp_shape)
		radius = geom_vals['radius']
		if np.random.random(1) < prob_alpha:
			color = (0,0,0,0)
		else:
			if np.random.random(1) < prob_palette:
				color = choice(palette)
			else:
				color = tuple(random_rgba())
		#print geom_vals
		centers = geom_vals['center'] + geom_vals['center2']
		#print centers
		#draw line
		draw.line(centers, fill = color, width = radius)
		#draw circle caps
		c1 = np.asarray(geom_vals['center'])
		c2 = np.asarray(geom_vals['center2'])
		if radius == 1:
			return 0
		newrad = max(1,radius//2)
		draw.ellipse((c1 - newrad).tolist() + (c1 + newrad).tolist(),fill=color,outline=color)
		draw.ellipse((c2 - newrad).tolist() + (c2 + newrad).tolist(),fill=color,outline=color)
	return 0


def extract_palette(image):
	carray = np.asarray(image)
	coldict = dict()
	for i in range(carray.shape[0]):
		for j in range(carray.shape[1]):
			color = carray[i,j,:].tolist()
			if color[3] == 255:
				coldict[str(color)] = color
	if len(coldict) == 0:
		coldict['a'] = [0,0,0,255]
	return coldict

def randint_bounds(range_vals):
	#print range_vals
	return np.random.randint(range_vals[1],range_vals[0])
#this may be a way to save memory while slightly increasing 
#computation time; allows for more flexibility when processing images
#deprecating shift amplitudes for boundary-detection
#added bilinear resampling to more properly emulate a diagonal stroke
def random_transform_image(img,rot_sd = 16,scale_range = [0.6,1.3],shift_amplitudes = [-45,45]):
	scale = np.random.uniform(scale_range[0],scale_range[1])
	angle = np.random.normal(0,rot_sd)
	shp = img.size
	new_shp = [int(x*scale) for x in shp]
	xdiff = new_shp[0] - shp[0]
	ydiff = new_shp[1] - shp[1]
	x2 = xdiff//2
	y2 = ydiff//2
	#rotate & scale
	#offset to center the new scaling (important for rotating)
	sca = img.resize(new_shp)
	rot = sca.rotate(angle,expand=1,resample=2)
	#redo shape parameters because of rotation expansion
	new_shp = rot.size
	xdiff = new_shp[0] - shp[0]
	ydiff = new_shp[1] - shp[1]
	x2 = xdiff//2
	y2 = ydiff//2
	#check image boundaries
	bounds = get_drawing_bounds(rot)
	range0 = (max( 5,bounds[0][0]-5+x2),min(-5,195 - bounds[0][1]-x2))
	range1 = (max( 5,bounds[1][0]-5+x2),min(-5,195 - bounds[1][1]-y2))
	#print bounds
	#print range0
	#print range1
	shift_values = [randint_bounds(x) for x in [range0,range1]]
	#print shift_values
	#shift_values = [np.random.randint(shift_amplitudes[0],shift_amplitudes[1]) for _ in range(2)]
	shi = rot.offset(shift_values[0],shift_values[1])
	cro = shi.crop((xdiff//2,ydiff//2,new_shp[0]-xdiff//2,new_shp[0]-ydiff//2)).resize((shp))
	return cro
