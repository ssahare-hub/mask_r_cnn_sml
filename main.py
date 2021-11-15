# %%
import os
import sys
sys.path.append("MaskRCNN")
import random
import math
import re
import time
import numpy as np
import cv2
import skimage
import matplotlib
import matplotlib.pyplot as plt

from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log

# from tqdm.notebook import tqdm
from tqdm.auto import tqdm
from tqdm.keras import TqdmCallback

import warnings
# warnings.filterwarnings("ignore")

# %%

# Root directory of the project
ROOT_DIR = os.getcwd()

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Directory to load source dataset
DATA_DIR = os.path.join(ROOT_DIR, "processed_data\\leftImg8bit")

# Directory to load groundtruth dataset
MASK_DIR = os.path.join(ROOT_DIR, "processed_data\\gtFine")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "MaskRCNN\\mask_rcnn_coco.h5")

# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

TRAINING = True
subset = ''

# %%
class CityscapeConfig(Config):
    """Configuration for training on the cityscape dataset.
    Derives from the base Config class and overrides values specific
    to the cityscape dataset.
    """
    # Give the configuration a recognizable name
    NAME = "cityscape"

    # We use a GPU with 12GB memory.
    GPU_COUNT = 1
    
    # 8
    IMAGES_PER_GPU = 1

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 6000

    # Number of validation steRPNps to run at the end of every training epoch.
    VALIDATION_STEPS = 50

    # Backbone network architecture
    # Supported values are: resnet50, resnet101
    BACKBONE = "resnet50"

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # background + 1 shapes

    # Input image resing
    IMAGE_RESIZE_MODE = "square"
    IMAGE_MIN_DIM = 800
    IMAGE_MAX_DIM = 1024

    # Learning rate and momentum
    LEARNING_RATE = 0.01
config = CityscapeConfig()

# %%
def get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    Change the default size attribute to control the size
    of rendered images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
    return ax


# %%
class CityscapeDataset(utils.Dataset):
    '''
    load_shapes()
    load_image()
    load_mask()
    '''

    def __init__(self, subset, max_images=0):
        super(CityscapeDataset, self).__init__(self)
        self.subset = subset
        self.max_images = max_images

    def load_shapes(self):
        """
        subset: "train"/"val"
        image_id: use index to distinguish the images.
        gt_id: ground truth(mask) id.
        height, width: the size of the images.
        path: directory to load the images.
        """
        # Add classes you want to train
        # self.add_class("cityscape", 1, "sidewalk")
        # self.add_class("cityscape", 2, "person")
        # self.add_class("cityscape", 3, "rider")
        self.add_class("cityscape", 1, "car")
        # self.add_class("cityscape", 5, "truck")
        # self.add_class("cityscape", 6, "bus")

        # Add images
        image_dir = "{}/{}".format(DATA_DIR, self.subset)
        image_ids = os.listdir(image_dir)
        if self.max_images != 0:
            image_ids = image_ids[:self.max_images]
        
        for index, item in tqdm(enumerate(image_ids), desc='preparing dataset'):
            temp_image_path = "{}/{}".format(image_dir, item)
            temp_image_size = skimage.io.imread(temp_image_path).shape
            self.add_image("cityscape", image_id=index, gt_id=os.path.splitext(item)[0],
                            height=temp_image_size[0], width=temp_image_size[1],
                            path=temp_image_path)


    def load_image(self, image_id):
        """Load images according to the given image ID."""
        info = self.image_info[image_id]
        image = skimage.io.imread(info['path'])
        return image

    def image_reference(self, image_id):
        """Return the shapes data of the image."""
        info = self.image_info[image_id]
        if info["source"] == "cityscape":
            return info["cityscape"]
        else:
            super(self.__class__).image_reference(self, image_id)

    def load_mask(self, image_id):
        """Load instance masks of the given image ID.
        count: the number of masks in each image.
        class_id: the first letter of each mask file's name.
        """
        info = self.image_info[image_id]
        gt_id = info['gt_id']
        # cityscapes = info["cityscape"]
        mask_dir = "{}\\{}\\{}".format(MASK_DIR, self.subset, gt_id)
        masks_list = os.listdir(mask_dir)
        count = len(masks_list)
        mask = np.zeros([info['height'], info['width'], count])
        class_ids = []

        for index, item in enumerate(masks_list):
            temp_mask_path = "{}\\{}".format(mask_dir, item)
            tmp_mask = 255 - skimage.io.imread(temp_mask_path)[:, :, np.newaxis]
            # print(item, tmp_mask.shape)
            mask[:, :, index:index+1] = tmp_mask
            class_ids.append(1)

        # Handle occlusions
        occlusion = np.logical_not(mask[:, :, -1]).astype(np.uint8)
        for i in range(count-2, -1, -1):
            mask[:, :, i] = mask[:, :, i] * occlusion
            occlusion = np.logical_and(occlusion, np.logical_not(mask[:, :, i]))
            
        # class_ids = np.array([self.class_names.index(c[0]) for c in cityscapes])
            
        return mask, np.array(class_ids, dtype=np.uint8)


# %%
if TRAINING:
    # Training dataset
    dataset_train = CityscapeDataset("train")
    dataset_train.load_shapes()
    dataset_train.prepare()

# Validation dataset
dataset_val = CityscapeDataset("val")
dataset_val.load_shapes()
dataset_val.prepare()

# %%
# Inspect the dataset
print("Image Count: {}".format(len(dataset_train.image_ids)))
print("Class Count: {}".format(dataset_train.num_classes))
for i, info in enumerate(dataset_train.class_info):
    print("{:3}. {:50}".format(i, info['name']))


# %%
# Create model in training mode
model = modellib.MaskRCNN(mode="training", config=config,
                          model_dir=MODEL_DIR)

# Which weights to start with?
init_with = "coco"  # imagenet, coco, or last

if init_with == "imagenet":
    model.load_weights(model.get_imagenet_weights(), by_name=True)
elif init_with == "coco":
    # Load weights trained on MS COCO, but skip layers that
    # are different due to the different number of classes
    # See README for instructions to download the COCO weights
    model.load_weights(COCO_MODEL_PATH, by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                                "mrcnn_bbox", "mrcnn_mask"])
elif init_with == "last":
    # Load the last model you trained and continue training
    model.load_weights(model.find_last(), by_name=True)


# %%
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    if TRAINING:
        # tqdm bar for tracking training progress
        callback = TqdmCallback()
        callback.display()
        
        # Train the head branches
        # Passing layers="heads" freezes all layers except the head
        # layers. You can also pass a regular expression to select
        # which layers to train by name pattern.
        # model.train(dataset_train, dataset_val,
        #             learning_rate=config.LEARNING_RATE,
        #             epochs=1,
        #             layers='heads')

        # Fine tune all layers
        # Passing layers="all" trains all layers. You can also
        # pass a regular expression to select which layers to
        # train by name pattern.
        # learning_rate = 0.01
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=3,
                    layers="all",
                    custom_callbacks = [callback],
                    verbose=0
                    )

        callback = TqdmCallback()
        callback.display()
        # learning_rate = 0.001
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE / 10,
                    epochs=1,
                    layers="all",
                    custom_callbacks = [callback],
                    verbose=0
                    )

        # Save weights
        # Typically not needed because callbacks save after every epoch
        # Uncomment to save manually
        # model_path = os.path.join(MODEL_DIR, "mask_rcnn_shapes.h5")
        # model.keras_model.save_weights(model_path)

