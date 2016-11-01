from distort import distortion
from distort import add_imperfections
import cv2
import constants as co
from constants import urand 
import random
import numpy as np
from export_filenames_to_postgres import BASIC_IMAGE_CLASSES, valid_names
from export_filenames_to_postgres import NOFLIP_IMAGE_CLASSES
from copy import copy

class distortion_wheel:
    def __init__(self,xdim,ydim):
        """remember xdim and ydim are flipped >_<"""
        self.warp_distort = distortion(xdim,ydim)
        self.wave_distort = distortion(xdim,ydim)
        self.scale_distort = distortion(xdim,ydim)
        self.rot_distort = distortion(xdim,ydim)
        self.displace = distortion(xdim,ydim)
        self.make_priorities()
        self.initialize_distortions()
    def set_warp(self):
        self.warp_distort.create_sinusoidal_warp(
            urand(co.warp_ax),
            urand(co.warp_ay),
            urand(co.warp_perx),
            urand(co.warp_pery),
            urand(co.warp_phax),
            urand(co.warp_phay)
            )
    def set_wave(self):
        self.wave_distort.create_sinusoidal_wave(
            urand(co.wave_ax),
            urand(co.wave_ay),
            urand(co.wave_perx),
            urand(co.wave_pery),
            urand(co.wave_phax),
            urand(co.wave_phay)
            )
    def set_tint(self):
        self.tint = [urand(co.rgb_shift) for _ in range(3)]
    def set_scale(self):
        self.scx = urand(co.scale_x)
        self.scy = urand(co.scale_y)
        self.scale_distort.calculate_scale(
            (self.scx,self.scy),
            offset=(
                urand(co.scale_x_offset),
                urand(co.scale_y_offset)
                )
            )

    def set_rotation(self):
        self.rot_distort.calculate_rotation(
            urand(co.rot_theta),
            offset = (
                urand(co.rot_offset_x),
                urand(co.rot_offset_y)
                )
            )

    def set_offset(self):
        self.displace.create_affine(
            1.,
            1.,
            0,
            0,
            urand(co.x_offset),
            urand(co.y_offset)
            )
    def initialize_distortions(self):
        self.set_warp()
        self.set_wave()
        self.set_scale()
        self.set_rotation()
        self.set_offset()
        self.set_tint()
        self.make_priorities()
    def make_priorities(self):
        self.wav_priority = urand(co.wave)
        self.warp_priority = urand(co.warp)
        self.affine_priority = urand(co.affine)
        self.maxv = max(self.wav_priority,self.warp_priority,self.affine_priority)
        self.minv = min(self.wav_priority,self.warp_priority,self.affine_priority)

    def rotate_values(self, num_distorts=1):
        #not particularly safe to use exec(), but should be fine
        distort_list = ['self.set_' + x + '()' for x in \
                            ['scale','wave','warp','offset','rotation','tint']]
        funcs = random.sample(distort_list,num_distorts)
        for func in funcs:
            exec(func)
        self.make_priorities()
        
    def smart_crop(self, image, to=(180, 180)):
        """will shrink image slightly so that it is free to shift around more"""
        shp = image.shape
        small = cv2.resize(image, to)
        new = np.zeros(shp, np.uint8)
        diff = (shp[0]-to[0], shp[1]-to[1])
        new[diff[0]//2:(shp[0]-diff[0]//2),diff[1]//2:(shp[1]-diff[1]//2)] = small
        return new

    def smart_displacement(self, image):
        """will not allow image to be shifted beyond its borders"""
        vsums = np.sum(image, (1,2))
        hsums = np.sum(image, (0,2))
        try:
            hvalid = np.where(hsums > 0.0)
            vvalid = np.where(vsums > 0.0)
            hbounds = (np.min(hvalid), np.max(hvalid))
            vbounds = (np.min(vvalid), np.max(vvalid))
            shp = image.shape
            hrange = (-hbounds[0], shp[0] - hbounds[1])
            vrange = (-vbounds[0], shp[1] - vbounds[1])
            #generate random rolls
            hshift = np.random.randint(hrange[0], hrange[1])
            vshift = np.random.randint(vrange[0], vrange[1])
            #roll it
            image = np.roll(image, hshift, axis=1)
            image = np.roll(image, vshift, axis=0)
        except ValueError:
            print( "image cannot be properly manipulated..."\
                "writing to erroneous_image.png")
            cv2.imwrite('/home/max/workspace/Sketch2/erroneous_image.png', image)
        return image

    def process_image(self, image, cls=None, smart=False):
        """processes image using distortions and some strokes"""
        stroke_first = np.random.random() < co.stroke_priority        
        #handle tint
        for j, tint in enumerate(self.tint):
            #will not illuminate 0-alpha tiles
            image[:,:,j] = np.uint8((image[:,:,3] > 0) *
                np.maximum(0,np.minimum(255,
                                        np.uint(image[:,:,j]) + tint
                                        )))
        max_strokes = co.max_strokes
        if smart:
            image = self.smart_crop(image)
        #reduces too many strokes on simple images
        if cls in BASIC_IMAGE_CLASSES:
            max_strokes = 1
        stroke_kwargs = copy(co.stroke_kwargs)
        stroke_kwargs['max_num_imps'] = max_strokes
        if stroke_first and max_strokes > 0:
            add_imperfections(image, **co.stroke_kwargs)
        #now do distortions
        if self.wav_priority == self.maxv:
            image = self.wave_distort.process_image(image)
            if self.warp_priority == self.minv:
                if (self.scx + self.scy)/2 > 1:
                    image = self.scale_distort.process_image(image)
                    image = self.rot_distort.process_image(image)
                else:
                    image = self.rot_distort.process_image(image)
                    image = self.scale_distort.process_image(image)
                image = self.warp_distort.process_image(image)
            else:
                image = self.warp_distort.process_image(image)            
                if (self.scx + self.scy)/2 > 1:
                    image = self.scale_distort.process_image(image)
                    image = self.rot_distort.process_image(image)
                else:
                    image = self.rot_distort.process_image(image)
                    image = self.scale_distort.process_image(image)
        elif self.warp_priority == self.maxv:
            image = self.warp_distort.process_image(image)
            if self.wav_priority == self.minv:
                if (self.scx + self.scy)/2 > 1:
                    image = self.scale_distort.process_image(image)
                    image = self.rot_distort.process_image(image)
                else:
                    image = self.rot_distort.process_image(image)
                    image = self.scale_distort.process_image(image)
                image = self.wave_distort.process_image(image)
            else:
                image = self.wave_distort.process_image(image)            
                if (self.scx + self.scy)/2 > 1:
                    image = self.scale_distort.process_image(image)
                    image = self.rot_distort.process_image(image)
                else:
                    image = self.rot_distort.process_image(image)
                    image = self.scale_distort.process_image(image)
        else:
            if (self.scx + self.scy)/2 > 1:
                image = self.scale_distort.process_image(image)
                image = self.rot_distort.process_image(image)
            else:
                image = self.rot_distort.process_image(image)
                image = self.scale_distort.process_image(image)
            if self.wav_priority == self.minv:
                image = self.warp_distort.process_image(image)
                image = self.wave_distort.process_image(image)
            else:
                image = self.wave_distort.process_image(image)
                image = self.warp_distort.process_image(image)
        #displacement
        if not smart:
            image = self.displace.process_image(image)
        else:
            image = self.smart_displacement(image)
        if (not stroke_first) and max_strokes > 0 :
            add_imperfections(image, **stroke_kwargs)
        if np.random.random() < co.flip_chance and cls not in NOFLIP_IMAGE_CLASSES:
            image = np.fliplr(image)
        return image
