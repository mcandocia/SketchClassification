#all of these are lower/upper bounds for a uniform distribution
from numpy import pi
import random
from copy import copy

warp_ax = (-4,7)
warp_ay = (-4,7)
warp_perx = (90,700)
warp_pery = (90,700)
warp_phax = (-100,100)
warp_phay = (-100,100)

wave_ax = (-4,7)
wave_ay = (-4,7)
wave_perx = (90,700)
wave_pery = (90,700)
wave_phax = (-190,190)
wave_phay = (-190,190)

rot_theta = (-19*pi/180.,19*pi/180.)
rot_offset_x = [-1,1]
rot_offset_y = [-1,1]

scale_x = [0.84,1.18]
scale_y = [0.84,1.18]
scale_x_offset = [-1,1]
scale_y_offset = [-1,1]

x_offset = [-10,10]
y_offset = [-10,10]

flip_chance = 0.45

#if tinting is used
rgb_shift = [-12,12]

#determines priority distribution for mappings

wave = (0,1)
warp = (0,1)
affine = (0,1)

#params for strokes
stroke_priority = 0.5#probability for stroke before distortion
max_strokes = 2
stroke_alpha_prob = 0.8
stroke_shape  = {'length_mean':12, 'length_sd':4, 'radius_mean':3, 'radius_sd':3, 'max_radius':7}
prob_palette=0.8#chance that borrows from palette of source image
stroke_kwargs = {'imp_shape':stroke_shape, 'prob_alpha':stroke_alpha_prob,
                 'prob_palette':prob_palette,'max_num_imps':max_strokes}

#used for processing these intervals
def urand(tup):
    return random.uniform(tup[0],tup[1])
