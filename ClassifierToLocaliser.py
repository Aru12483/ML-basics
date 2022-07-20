import tensorflow as tf
import numpy as np
tf.random.set_seed(100)

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Activation, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import  EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import SGD
from keras.utils.vis_utils import plot_model
from keras.models import load_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dropout
from keras import backend as K 
from tensorflow.keras.utils import Sequence

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import cv2
import random
import os
import glob

import urllib
import tarfile
from zipfile import ZipFile
import scipy.io
from PIL import Image, ImageDraw, ImageEnhance

import imgaug as ia
ia.seed(1)

#%matplotlib inline
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from imgaug import augmenters as iaa 


import imageio
import pandas as pd
import numpy as np
import re

import shutil
#s!wget -O CALTECH.zip https://data.caltech.edu/tindfiles/serve/e41f5188-0b32-41fa-801b-d1e840915e80/


# Extract all the contents of zip file in current directory
with ZipFile('CALTECH.zip', 'r') as zipObj:
   zipObj.extractall()

# creating a mat parser
def extract_mat_contents(annot_directory,img_dir):
  mat = scipy.io.loadmat(annot_directory)
  height,width = cv2.imread(img_dir).shape[:2]
   # Get the bounding box co-ordinates
  x1, y2, y1, x2 = tuple(map(tuple, mat['box_coord']))[0]
  class_name = image_dir.split('/')[2]
  filename = '/'.join(image_dir.split('/')[-2:])
  return filename,  width, height, class_name, x1,y1,x2,y2


# Function to convert MAT files to CSV
def mat_to_csv(annot_directory, image_directory, classes_folders):

  # List containing all our attributes regarding each image
  mat_list = []

  # We loop our each class and its labels one by one to preprocess and augment 
  for class_folder in classes_folders:

    # Set our images and annotations directory
    image_dir = os.path.join(image_directory, class_folder)
    annot_dir = os.path.join(annot_directory, class_folder) 

    # Get each file in the image and annotation directory
    mat_files = sorted(os.listdir(annot_dir))
    img_files = sorted(os.listdir(image_dir))

    # Loop over each of the image and its label
    for mat, image_file in zip(mat_files, img_files):
      
      # Full mat path
      mat_path = os.path.join(annot_dir, mat)

      # Full path Image
      img_path = os.path.join(image_dir, image_file)

      # Get Attributes for each image 
      value = extract_mat_contents(mat_path, img_path)

      # Append the attributes to the mat_list
      mat_list.append(value)

  # Columns for Pandas DataFrame
  column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin','xmax', 'ymax']

   # Create the DataFrame from mat_list
  mat_df = pd.DataFrame(mat_list, columns=column_name)

  # Return the dataframe
  return mat_df

# The Classes we will use for our training
classes_list = sorted(['butterfly',  'cougar_face', 'elephant'])


# Set our images and annotations directory
image_directory = '/content/caltech-101/101_ObjectCategories.tar.gz'
annot_directory = '/content/caltech-101/Annotations.tar'

# Run the function to convert all the MAT files to a Pandas DataFrame
labels_df = mat_to_csv(annot_directory, image_directory, classes_list)

# Saving the Pandas DataFrame as CSV File
labels_df.to_csv(('labels.csv'), index=None)

# Function to convert bounding box image into DataFrame 
def bounding_boxes_to_df(bounding_boxes_object):

    # Convert Bounding Boxes Object to Array
    bounding_boxes_array = bounding_boxes_object.to_xyxy_array()
    
    # Convert the array into DataFrame
    df_bounding_boxes = pd.DataFrame(bounding_boxes_array, 
                                     columns=['xmin', 'ymin', 'xmax', 'ymax'])
    
    # Return the DataFrame
    return df_bounding_boxes