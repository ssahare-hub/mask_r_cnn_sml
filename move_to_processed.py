import os
import re
import shutil
SOURCE_DIR = './data/leftImg8bit'
subpaths = os.listdir(SOURCE_DIR)
OUT_DIR = './processed_data/leftImg8bit/'
pattern = r'.+.png$'

for subpath in subpaths:
    # join the './gtFine' for walk dir
    sub_path = os.path.join(SOURCE_DIR, subpath)
    
    # create the output dir
    if not os.path.exists(os.path.join(OUT_DIR, subpath)):
        os.mkdir(os.path.join(OUT_DIR, subpath))

    temp_destination = os.path.join(OUT_DIR, subpath)

    for dirpath, dirnames, filenames in os.walk(sub_path):
        for filename in filenames:
            if re.match(pattern, filename):
                srcimage = os.path.join(dirpath, filename)
                shutil.move(srcimage, temp_destination)
                print("Successfully moved {} -> {}".format(srcimage, temp_destination))