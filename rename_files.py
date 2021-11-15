import os
import re
import shutil
# do this for all gtFine/train gtFine/val leftImg8bit/train leftImg8bit/val leftImg8bit/test
path = './processed_data/leftImg8bit/val'
files = os.listdir(path)

counter = 0
for file in files:
    try:
        newname = re.findall(r'(.+)_.+\.png$', file)[0]
        newname += '.png'
    except Exception as e:
        print(e)
        continue
    print(newname)
    src_file = os.path.join(path, file)
    dst_file = os.path.join(path, newname)
    # os.rename(src_file, dst_file)
    print("Successfully rename {} -> {}".format(src_file, dst_file))
    counter += 1

print(counter)