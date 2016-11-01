import csv
import psycopg2
from numpy import random
import dbinfo
import re
import os

train_proportion = 0.7
test_proportion = 0.15
validation_proportion = 0.15

test_threshold = train_proportion + test_proportion

MAX_BATCH_SIZE = 5000

valid_names = ['cat','dog','cuban flag','banana','horse','tree','smiley face',
               'rainbow','flower','car','circle','square','triangle','fish',
               'bee',
               'star','turtle','bird','butterfly','snowman','heart','cube',
               'american flag','puerto rican flag','tanzanian flag',
               'nigerian flag','canadian flag','russian flag','chinese flag',
               'japanese flag','south korean flag','jamaican flag',
               'ukrainian flag',
               'chilean flag','israeli flag','laotian flag','swedish flag',
               'german flag','french flag','stick figure','rectangle',
               'mexican flag','british flag','indian flag','spiral','trebuchet',
               'pumpkin','mushroom','ghost','pineapple','wizard',
               'witch','ninja','dragon','pirate','santa claus','elf']

BASIC_IMAGES = ['banana','smiley face','circle','square','triangle','star',
                'heart','cube','stick figure','rectangle','spiral']

BASIC_IMAGE_CLASSES = [i for i, x in enumerate(valid_names) 
                       if x in BASIC_IMAGES]


NOFLIP_IMAGES = ['cuban flag','american flag','puerto rican flag',
                 'tanzanian flag','chinese flag','chilean flag','swedish flag']

NOFLIP_IMAGE_CLASSES = [i for i, x in enumerate(valid_names) 
                        if x in NOFLIP_IMAGES]

def main():
    conn = psycopg2.connect("dbname=%s user=%s password=%s host=%s port=%s" %
                            (dbinfo.dbname,dbinfo.user,dbinfo.password,
                             dbinfo.host,dbinfo.port))
    print 'connected to database'
    cur = conn.cursor()
    cur.execute("DROP TABLE IF EXISTS images;")
    cur.execute("CREATE TABLE images(class INTEGER, filename text,role integer, random_index integer);")

    regex = '__.*|'.join(valid_names)+'__.*'
    files_list = os.listdir('/home/max/Pictures/blog_project_images/images/')
    files_list = [x for x in files_list if re.match('.*png$',x)]
    files_list = [x for x in files_list if re.match(regex,x)]
    data_rows = []
    for i, f in enumerate(files_list):
        object_class = re.sub('__.*','',f).lower()
        object_class_id = valid_names.index(object_class)
        data_rows.append((object_class_id, f))
        if not (i+1) % MAX_BATCH_SIZE:
            store_data(data_rows, cur)
            data_rows = []
    if (i+1) % MAX_BATCH_SIZE:
        store_data(data_rows, cur)
    print 'stored data'
    conn.commit()
    conn.close()

def rand_cat():
    x = random.random()
    if x < train_proportion:
        return [1]
    elif x < test_threshold:
        return [2]
    else:
        return [3]

maxindex = pow(2,22)
def rand_index():
    return [round(random.random() * maxindex)]

def make_path(name,category):
    return '/home/max/workspace/StateFarmDistract/train/%s/%s' % (category,name)

def store_data(data,cursor):
    ldata = [[row[0],row[1]] for row in data]
    reformatted_data = [list(row) + rand_cat() + rand_index() for row in ldata]
    argstring = ','.join(cursor.mogrify("(%s,%s,%s,%s)",x) \
                         for x in reformatted_data)
    cursor.execute("INSERT INTO images VALUES " + argstring)
    print "inserted entries into db"
    
if __name__=='__main__':
    main()
