from PIL import Image
import numpy as np
import os
import dbinfo
import psycopg2
import sys
import re

new_directory = '/home/max/workspace/Sketch2/transformed/'
image_directory = '/home/max/Pictures/blog_project_images/images/'
regexp = re.compile('\.png')

#original dims and scales currently not used
original_dims = (400, 400)
target_dims = (200,200)

scales = (target_dims[0]/float(original_dims[0]), target_dims[1]/float(original_dims[1]))

def main():
    conn = psycopg2.connect("dbname=%s user=%s password=%s host=%s port=%s" %
(dbinfo.dbname,dbinfo.user,dbinfo.password,dbinfo.host,dbinfo.port))
    c = conn.cursor()
    c.execute("SELECT filename FROM images;")
    if not os.path.exists(new_directory):
        print 'making new directories'
        os.makedirs(new_directory)
    else:
        print "Directories already exist...continuing"
    while True:
        res = c.fetchmany(1000)
        if len(res) == 0:
            break
        for fn in res:
            scale_image(fn[0])
        sys.stdout.write('.')
        sys.stdout.flush()
    print ''
    print 'finished transforming images'
    c.execute("DROP TABLE IF EXISTS images_transformed;")
    c.execute("CREATE TABLE images_transformed AS (SELECT * FROM images);")
    c.execute("UPDATE images_transformed SET filename=regexp_replace(filename,'\.png','train_transformed.png');")
    conn.commit()
    print 'created images_transformed table'
    conn.close()
    return 0

def scale_image(filename):
    new_filename = new_directory + regexp.sub('train_transformed.png', filename)
    dst_im = Image.new("RGBA",target_dims,(0,0,0))
    try:
        base_image = Image.open(image_directory + filename)
    except:
        print filename
        raise
    bi = base_image.resize(target_dims)
    dst_im.paste(bi,((0,0)))
    dst_im.save(new_filename)

if __name__=='__main__':
    main()
