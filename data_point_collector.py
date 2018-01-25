# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 07:20:42 2017

@author: pasca
"""

import os
from six.moves import cPickle as pickle
from tensorflow.python.platform import gfile
import glob
import re
import collections




#data_dir = 'data/'
def read_datasets(data_dir):
    pickle_filename = "data_point_names.pickle"
    pickle_filepath = os.path.join(data_dir, pickle_filename)
    if not os.path.exists(pickle_filepath):
        data_point_names = get_data_point_names(data_dir)
        print ("Pickling ...")
        with open(pickle_filepath, 'wb') as f:
            pickle.dump(data_point_names, f, pickle.HIGHEST_PROTOCOL)
    else:
        print ("Found pickle file!")
        with open(pickle_filepath, 'rb') as f:
            data_point_names = pickle.load(f)

    return data_point_names['training'], data_point_names['validation']


#data_dir = data/
#training set directory: data/training/
#validation set directory: data/validation/
#camera_images/
#gazemap_images/
#10_342.jpg

def get_data_point_names(data_dir):
    if not gfile.Exists(data_dir):
    #if not os.path.isdir(data_dir):
        print("Data directory '" + data_dir + "' not found.")
        return None
    directories = ['training', 'validation']
    data_point_names = {}

    for directory in directories:
        camera_list = []
        data_point_names[directory] = []
        name_pattern = os.path.join(data_dir, directory, 
                                    "camera_images", '*.' + 'jpg')
        camera_list.extend(glob.glob(name_pattern))

        if not camera_list:
            print('No files found')
        else:
            for camera_full_path in camera_list:
                file_id = re.search('([0-9]+_[0-9]+).jpg', 
                                      camera_full_path).group(1)
                gazemap_full_path = os.path.join(data_dir, directory, 
                                                 "gazemap_images", file_id+'.jpg')
                if os.path.exists(gazemap_full_path):
                    data_point_names[directory].append(file_id)
                else:
                    print("Annotation file not found for %s - Skipping" % file_id)
        no_of_images = len(data_point_names[directory])
        print ('No. of %s files: %d' % (directory, no_of_images))

    return data_point_names


def read_datasets_in_sequences(data_dir):
    pickle_filename = "data_point_names_in_sequences.pickle"
    pickle_filepath = os.path.join(data_dir, pickle_filename)
    if not os.path.exists(pickle_filepath):
        data_point_names_in_sequences = get_data_point_names_in_sequences(data_dir)
        print ("Pickling ...")
        with open(pickle_filepath, 'wb') as f:
            pickle.dump(data_point_names_in_sequences, f, pickle.HIGHEST_PROTOCOL)
    else:
        print ("Found pickle file!")
        with open(pickle_filepath, 'rb') as f:
            data_point_names_in_sequences = pickle.load(f)

    return (data_point_names_in_sequences['training'], 
            data_point_names_in_sequences['validation'])

def get_data_point_names_in_sequences(data_dir):
    if not gfile.Exists(data_dir):
    #if not os.path.isdir(data_dir):
        print("Data directory '" + data_dir + "' not found.")
        return None
    directories = ['training', 'validation']
    data_point_names_in_sequences = {}

    for directory in directories:
        camera_list = []
        video_frames_dict = collections.defaultdict(list)
        name_pattern = os.path.join(data_dir, directory, 
                                    "camera_images", '*.' + 'jpg')
        camera_list.extend(glob.glob(name_pattern))
        camera_list.sort()

        if not camera_list:
            print('No files found')
        else:
            for camera_full_path in camera_list:
                file_id = re.search('([0-9]+_[0-9]+).jpg', 
                                      camera_full_path).group(1)
                gazemap_full_path = os.path.join(data_dir, directory, 
                                                 "gazemap_images", file_id+'.jpg')
                if os.path.exists(gazemap_full_path):
                    video_id = file_id.split('_')[0]
                    video_frames_dict[video_id].append(file_id)
                    #data_point_names[directory].append(file_id)
                else:
                    print("Annotation file not found for %s - Skipping" % file_id)
            data_point_names_in_sequences[directory] = \
            list(video_frames_dict.values())
        no_of_videos = len(data_point_names_in_sequences[directory])
        print ('No. of %s videos: %d' % (directory, no_of_videos))

    return data_point_names_in_sequences

def keep_only_videos(data_point_names_in_sequences, video_list):
    filtered = []
    for seq in data_point_names_in_sequences:
        for target in video_list:
            if seq[0].split('_')[0] == target:
                filtered.append(seq)
                break
    return filtered
    
    
    
    