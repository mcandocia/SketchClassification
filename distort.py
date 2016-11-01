import cv2
import numpy as np
from scipy.interpolate import griddata
from PIL import Image, ImageDraw
from skimage import draw as skd

#class that contains basic grid info and then can generate wacky 
#distortions that can be used on images arbitrarily
#one distortion can be saved at a time but used indefinitely
#note that x is vertical and y is horizontal
class distortion:
    #supports linear and cubic interpolation
    def __init__(self,xdim,ydim,method = 'linear'):
        xdm = xdim - 1
        ydm = ydim - 1
        self.xdim = xdim
        self.method = method
        self.ydim = ydim
        self.grid_x, self.grid_y = np.mgrid[0:xdm:(xdim*1j),0:ydm:(ydim*1j)]
        self.source = np.array([[i,j] for i,j in\
                                    zip(self.grid_x.flat,self.grid_y.flat)])
    def create_sinusoidal_warp(self,ax,ay,perx,pery,phax,phay):
        """strange warping that crosses between dimensions"""
        sinfunc = lambda x, y: (
            x + ax * np.sin(np.pi*2/pery*(y-phay)),
            y + ay * np.sin(np.pi*2/perx*(x-phax))
            )
        self.destination = [sinfunc(x,y) for x,y in self.source]
        self.calculate_transformation()
    def create_sinusoidal_wave(self,ax,ay,perx,pery,phax,phay):
        """warping that creates wave-like compressions and rarefactions"""
        sinfunc = lambda x, y: (
            x + ax * np.sin(np.pi*2/perx*(x-phax)),
            y + ay * np.sin(np.pi*2/pery*(y-phay))
            )
        self.destination = [sinfunc(x,y) for x,y in self.source]
        self.calculate_transformation()
    def create_affine(self,mx,my,mxy,myx,dx,dy):
        tfunc = lambda x, y: (mx * x + mxy * y + dx, my * y + myx * x + dy)
        self.destination = [tfunc(x,y) for x,y in self.source]
        self.calculate_transformation()
    def calculate_rotation(self,theta,center=None,offset = None):
        """rotates image; more useful than other rotations since most
        calculations are done initially rather than for each image"""
        if center==None:
            if offset == None:
                offset = (0,0)
            center = (self.xdim/2 + offset[0],self.ydim/2 + offset[1])
        xtrans = (np.cos(theta),np.sin(theta))
        ytrans = (-np.sin(theta),np.cos(theta))
        dx = center[0] - (xtrans[0] * center[0] + xtrans[1] * center[1])
        dy = center[1] - (ytrans[0] * center[0] + ytrans[1] * center[1])
        self.create_affine(xtrans[0],ytrans[1],xtrans[1],ytrans[0],dx,dy)
    def calculate_scale(self,scales,center=None,offset=None):
        if center==None:
            if offset == None:
                offset = (0,0)
            center = (self.xdim/2 + offset[0],self.ydim/2 + offset[1])
        dx = center[0] * (1-scales[0])
        dy = center[1] * (1-scales[1])
        self.create_affine(scales[0],scales[1],0,0,dx,dy)
    def calculate_transformation(self):
        self.grid_z = griddata(
            self.destination,
            self.source,
            (self.grid_x,self.grid_y),
            method=self.method
            )
        map_x = np.append([],[ar[:,1] for ar in self.grid_z]).\
            reshape(self.xdim,self.ydim)
        map_y = np.append([],[ar[:,0] for ar in self.grid_z]).\
            reshape(self.xdim,self.ydim)
        self.mx32 = map_x.astype('float32')
        self.my32 = map_y.astype('float32')
    def add_bad_stroke(self, img):
        """better for sketch-type images; adds random color to obscure
        part of image/make it misleading"""
        pass
    def process_image(self,img):
        if self.method=='linear':
            method = cv2.INTER_LINEAR
        else:
            method = cv2.INTER_CUBIC
        return cv2.remap(img,self.mx32,self.my32,method)

#functions borrowed from prepare_image_data.py

def random_rgb():
	return [np.random.randint(255) for _ in range(3)] 

def random_rgba():
	return np.asarray([np.random.randint(255) for _ in range(3)] + [255.])

def remove_out_of_bounds(overlay, h, w):
    o = overlay
    overlay = [(rr, cc, val) for (rr, cc, val) in zip(o[0], o[1], o[2]) if 
     rr >= 0 and rr < w and cc >= 0 and cc < h]
    return overlay
    
def draw_geometries(image, geometry1, geometry2, color):
    solids = geometry1['solid'] + geometry2['solid']
    antialiased = geometry1['antialiased'] + geometry2['antialiased']
    for solid in solids:
        bounded_solid = remove_out_of_bounds(solid, image.shape[0], image.shape[1])
        rr = [x[0] for x in bounded_solid]
        cc = [x[1] for x in bounded_solid]
        val = np.asarray([x[2] for x in bounded_solid])
        overlay_numpy_array(image, val, color, rr, 
                            cc)
    for aa in antialiased:
        bounded_aa = remove_out_of_bounds(aa, image.shape[0], image.shape[1])
        rr = [x[0] for x in bounded_aa]
        cc = [x[1] for x in bounded_aa]
        val = np.asarray([x[2] for x in bounded_aa])
        overlay_numpy_array(image, val, color, 
                            rr, cc)
    return 0

def stroke_to_circle_coordinates(center1, center2, radius):
    circle_geometries = []
    circle_antialiased = []
    for center in (center1, center2):
        rr, cc = skd.circle(center[0], center[1], radius)
        val = np.ones(len(rr))
        circle_geometries.append((rr, cc, val))
        aarr, aacc, aaval = skd.circle_perimeter_aa(center[0], center[1], radius)
        circle_antialiased.append((aarr, aacc, aaval))
    return {'solid':circle_geometries, 'antialiased':circle_antialiased}

def create_rectangle_from_coordinates(points):
    geometry = {}
    #solid component
    srr, scc = skd.polygon(np.array([p[0] for p in points]),
                           np.array([p[1] for p in points]))
    sval = np.ones(len(srr))
    geometry['solid'] = [[srr, scc, sval]]
    #4 linear components (these will just be overlayed altogether, later)
    aa_overlays = []
    for perm in ([0,1],[1,2],[2,3],[3,0]):
        p1 = points[perm[0]]
        p2 = points[perm[1]]
        rr, cc, val = skd.line_aa(p1[0],p1[1],p2[0],p2[1])
        aa_overlays.append((rr, cc, val))
    geometry['antialiased'] = aa_overlays
    return geometry

def overlay_numpy_array(image, val, color, rr, cc):
    """overlays val * color over the domain of rr and cc, broadcasted properly"""
    if len(val) == 0:
        #print 'all elements oob'
        return 1
    vals_expanded = np.broadcast_to(val, (len(color),len(val)))
    colors_expanded = np.asarray([[c for _ in val] 
                                  for c in color])
    color_vals = vals_expanded * colors_expanded
    color_vals = color_vals.T
    image_residual = image[rr,cc,:] * np.expand_dims((1-val),1)
    image[rr,cc,:] = np.int32(np.floor(np.add(image_residual, color_vals)))
    return 0

def stroke_to_rectangle_coordinates(center1, center2, width):
    """returns polygon val, rr, and cc; removes entries outside of bounds"""
    c1x = center1[0]
    c2x = center2[0]
    c1y = center1[1]
    c2y = center2[1]
    angle = np.arctan((c2y-c1y)/(c2x-c1x))
    new_angle = angle + np.pi/2
    points = []
    w2 = width/2
    for i, c in enumerate([center1, center2]):
        if i == 0:
            plusminus = [1, -1]
        else:
            plusminus = [-1, 1]
        for direction in plusminus:
            points.append((np.int(np.round(c[0] + 
                                           direction * w2 * np.cos(new_angle))), 
                           np.int(np.round(c[1] + 
                                           direction * w2 * np.sin(new_angle)))))
    #now actually do something with the points
    return create_rectangle_from_coordinates(points)
    
def stroke_geom_values(img_size,imp_shape):
	center = [np.random.randint(size) for size in img_size]
	angle = np.random.normal(0,np.pi)
	length = np.maximum(1,np.random.normal(imp_shape['length_mean'],imp_shape['length_sd']))
	center2 = [int(np.round(center[0] + np.cos(angle)*length)),
                   int(np.round(center[1] + np.sin(angle)*length))]
	radius = int(np.minimum(imp_shape['max_radius'],
                                np.maximum(1,np.random.normal(
                                    imp_shape['radius_mean'],
                                    imp_shape['radius_sd']))))
	return {'center':center, 'center2': center2, 'radius':radius}


#add strokes to image as imperfections to help generalize the algorithm
#use 
def add_imperfections(image, max_num_imps = 5, 
	imp_shape = {'length_mean':12,'length_sd':10,
                     'radius_mean':6,'radius_sd' : 4,'max_radius' : 18},
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
	size = image.shape

	for i in range(number_strokes):
		geom_vals = stroke_geom_values(size,imp_shape)
		radius = geom_vals['radius']
                #updated version uses random sample from source image
		if np.random.random(1) < prob_alpha:
			color = (0,0,0,0)
		else:
                    if np.random.random(1) < prob_palette:
                          color = image[np.random.randint(1,size[1]-1),
                                        np.random.randint(1,size[2]-1),
                                        :]
                          color[3] = 255
                    else:
                          color = tuple(random_rgba())
                color = np.asarray(color)
		#print geom_vals
		centers = geom_vals['center'] + geom_vals['center2']
		c1 = np.asarray(geom_vals['center'])
		c2 = np.asarray(geom_vals['center2'])
		#draw line
                rectangle_geometry = stroke_to_rectangle_coordinates(c1, c2, 
                                                                     radius * 2)
		#draw circle caps
                if radius > 1:
                    circle_geometry = stroke_to_circle_coordinates(c1, c2, 
                                                                   radius)
                else:
                          circle_geometry = {'solid':[], 'antialiased':[]}
                draw_geometries(image, rectangle_geometry, circle_geometry, color)
	return 0
