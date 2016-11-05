Python 2.7 dependencies: 
       * OpenCV2^
       * Psycopg2^
       * theano
       * numpy
       * sklearn^
       * matplotlib (for visualize_layers.py)
       * scipy^
       * skimage^

^ = not needed if you plan on using your own pre-processing and database system

How to configure software:

0. Set up a Postgresql Database, changing the info in dbinfo.py to match this. 

0.5. Set up theano. You will probably want to use a guide for this. Make sure         that you have GPU enabled under ~/.theanorc or whatever your config file 
   should be. floatX should be set to float32 for the GPU.
1. Have your RGBA PNG files in a directory. Currently class is detected by the 
   filename being in the structure of [classname]__[other text].png. The double
   underscore is necessary.
2. Configure export_filenames_to_postgres.py to match the image types you want. 
 * NOFLIP_IMAGES contains classes that do not have horizontal symmetry
 * BASIC_IMAGES contains classes that are so simple that fewer strokes will be 	 
   added as imperfections as to avoid too much confusion during training.
 * Change the beginning of main() if you plan on using a non-Postgres database
   to store image metadata
 * Change the string in files_list to the directory containing your images.

3. Configure transform_images.py
 * Change the directory of the original images and the transformed images.
 * Change the original and target dimensions if you do not want to convert
   a 400x400 image to 200x200
    
4. Configure prepare_image_data.py
 * xdim and ydim determine what the maximum image size distort_wheel.py will 
   use when transforming images before eventually cropping it to cropx and 
   cropy
 * WHEEL_MODIFICATIONS is a value that indicates how many different types of 
   transformations, both linear and nonlinear, will be changed between batches.
   It takes a little while for OpenCV2 to calculate these, so I have it at 1
   by default.
 
5. Configure constants.py
 * warp_* parameters are lateral wave distortions, and a*, per*, pha* indicate
   amplitude, period, and phase ranges
 * wave_* parameters are longitudinal wave distortions, with the same naming
   scheme as warp_*
 * rot_theta is the range for rotation, and a small offset range is included
 * scale_* is the range for scaling an image. You may try altering the original
   code in distort.py if you want to lock the x-scaling and y-scaling together
 * x_offset and y_offset are deprecated in favor of a displacement function 
   that shifts images to the extend of their alpha !=0 borders will not exit
   the image (until possible rotation or scaling)
 * flip_chance: probability that an image not belonging to NOFLIP_CLASSES will
   be flipped horizontally
 * rgb_shift: for alpha>0, RGB values will be independently shifted by some 
   value in this range. Will have a tinting effect. Will not result in values
   under 0 or over 255
 * stroke_priority: probability an imperfection stroke is made before
   distortions
 * max_strokes: most number of imperfection strokes for an image
 * stroke_alpha_prob: probability that a stroke will "erase" part of an image
   with RGBA = (0,0,0,0)
 * stroke_shape: parameters for stroke feature distribution
 * prob_palette: If stroke is not an "erase" stroke, what is the probability
   the color will be randomly selected from the palette of the original image
 
6. Configure an_00.py (or whatever you want to name it)
 * address_c should be changed to wherever you want to save your file
 * Other parameters should be changed depending on what you want your network
   architecture to be. full_neural_network.py should handle the algebra for
   this.

7. Configure full_neural_network.py
 * By default, each epoch will have its own pickle saved so you can track the
   network over each epoch (and retrieve error, learning rate, layer 
   parameters, etc.)
   - Change subdirectory = 'neural_pickle_history/' to a directory that 
     you want to store historical network data to, OR
   - Delete "save_network(epoch=True)" at the end of the epoch iteration loop
     (currently line 491)

8. Run an_00.py. If you are reloading it after it is stopped, it will load
   a network titled ...bnetwork...pickle, indicating the "best error" network.
   "inetwork" is a network created after the program is interrupted, and
   "fnetwork" is a network created once you've gone through all epochs. I 
   However, it will ONLY RELOAD IF YOU HAVE TRUE/T/True as a command argument
   (e.g., python an_00.py T), otherwise it will create a new network.
   Copying historical copies to the desired file is recommended if the last
   network saved isn't from the most recent epoch or you accidentally overwrote
   the network files by forgetting to use the command-line argument.
         
