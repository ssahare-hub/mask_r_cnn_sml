# mask_r_cnn_sml
SML Project

https://github.com/mcordts/cityscapesScripts
https://github.com/matterport/Mask_RCNN


# steps
- Download cityscapes dataset, specifically gtFine_trainvaltest and leftImg8bit_trainvaltest
- extract gtFine and leftImg8bit to data/
- run move_to_processed.py and you should get images in processed_data/leftImg8bit
- now toggle the commented code and run move_to_processed.py again and you should get json files in processed_data/gtFine
- move /processed_data/gtFine/test/ to  processed_data/
- run rename_files.py for each path ->  'gtFine/train' 'gtFine/val' 'leftImg8bit/train' 'leftImg8bit/val' 'leftImg8bit/test'
- at the end of these steps, you should have .json files inside processed_data/gtFine/* and .png in processed_data/leftImg8bit/*
- then add environment variable on your OS however it is, I only know windows, so for windows people
- to add an environment variable in powershell do this -> $Env:CITYSCAPES_DATASET='<path to processed_data/>'
- then run cityscapesScripts-master\cityscapesscripts\preparation\createTrainIdLabelImgs.py 
- you should have folders created inside processed_data/gtFine/*/ with one .png file per folder with weird black masks in white background
- replace tensorflow-gpu with tensorflow==1.13.1 and keras to keras==2.0.8
- install requirements.txt in the root dir
- scratch your head and figure out all the import errors
- pray to god and run main.py and hope your model starts training
- if it starts training, celebrate!!