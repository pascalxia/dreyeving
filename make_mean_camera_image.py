# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 19:07:38 2017

@author: pasca
"""


import os
import scipy.misc as misc
from tqdm import tqdm
import numpy as np
import ut
import feather
import argparse
import pandas as pd


import data_point_collector
import BatchDatasetReader


parser = argparse.ArgumentParser()

parser.add_argument('--data_dir', type=str,
                    default='data/')


args = parser.parse_args()


image_size = (576, 1024)



#set up data readers-------------------------------
train_data_points, valid_data_points = \
    data_point_collector.read_datasets(args.data_dir)
train_dataset_reader = \
    BatchDatasetReader.BatchDataset(args.data_dir+'training/',
                         train_data_points,
                         image_size=image_size)
    
#load all training camera_images
mean_camera = np.zeros(image_size+(3,))
batch_size = 10
n_batch = round(len(train_dataset_reader.data_point_names)/batch_size)
for i in tqdm(range(n_batch)):
    batch = train_dataset_reader.next_batch(batch_size)
    train_camera_images = train_dataset_reader.get_images(batch, image_size)
    mean_camera = mean_camera + np.mean(train_camera_images, axis=(0))

mean_camera = mean_camera/n_batch

mean_folder = args.data_dir+'training/camera_images/mean_frame/'
if not os.path.isdir(mean_folder):
    os.makedirs(mean_folder)

misc.imsave(mean_folder+'mean_camera_image.jpg', mean_camera)



