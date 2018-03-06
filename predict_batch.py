from __future__ import print_function

import data_point_collector
import BatchDatasetReader
import tensorflow as tf
import datetime
import scipy.misc as misc
import numpy as np
import pickle
from keras import backend as K
import pdb
import argparse
import ut
import os

from utils import getCoarse2FineModel
from keras.optimizers import Adam
import cv2
from tqdm import tqdm


#set flags--------------------------
parser = argparse.ArgumentParser()
ut.add_args_for_general(parser)
ut.add_args_for_inference(parser)
ut.add_args_for_training(parser)
ut.add_args_for_feature(parser)
ut.add_args_for_lstm(parser)
ut.add_args_for_evaluation(parser)


args = parser.parse_args()
ut.parse_for_general(args)
ut.parse_for_feature(args)


#set parameters-------------------
args.epsilon = 1e-12
args.image_size = (448, 448)
args.gaze_map_size = (448, 448)
args.display_size = (252, 448)
args.mean_frame_path = args.data_dir + 'training/camera_images/mean_frame/mean_camera_image.jpg'
args.output_map_size = (36, 64)


#set up session------------------
if args.gpu_memory_fraction is not None:
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = args.gpu_memory_fraction
    sess = tf.Session(config=config)
else:
    sess = tf.Session()
#assign session for Keras
K.set_session(sess)



#set up readout net-----------------
model = getCoarse2FineModel(summary=True)
opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(optimizer=opt,
              loss={'cropped_output': 'mse', 'full_fine_output': 'mse'},
              loss_weights={'cropped_output': 0.0, 'full_fine_output': 1.0})
              
if not os.path.isdir(args.logs_dir):
  os.makedirs(args.logs_dir)
model.save(args.logs_dir+'dreyeving_tf.h5')

    
#try to reload weights--------------------
if args.model_dir is not None:
    weight_path = args.model_dir + 'weights_iter_' + args.model_iteration + '.h5'
    model.load_weights(weight_path)
    print("Model restored...")
    




#set up data readers-------------------------------
_, valid_data_points = \
    data_point_collector.read_datasets_in_sequences(args.data_dir)
    
#pdb.set_trace()
    
validation_dataset_reader = \
    BatchDatasetReader.BatchDataset(args.data_dir+'validation/',
                         valid_data_points, 
                         args.image_size,
                         feature_name=args.feature_name,
                         annotation_threshold=args.annotation_threshold)
                         
                         
#train_dataset_reader = validation_dataset_reader


#set up summary writer
summary_writer = tf.summary.FileWriter(args.logs_dir)



#start training-------------------------
h = 112
w = 112

mean_frame = misc.imread(args.mean_frame_path)
mean_frame = cv2.resize(mean_frame, args.image_size)
#mean_frame = np.array([123.68, 116.79, 103.939])

def prepare_data(input_images_in_seqs, annotations_in_seqs, mean_frame):
    #input_images_in_seqs.shape=(?,16,448,448,3)
    #annotations_in_seqs.shape = (?,16,448,448,1)
    #mean_frame.shape = (448,448,3)
    
    x_raw = np.array([[f.astype(np.float32) - mean_frame for f in seq] for seq in input_images_in_seqs])
    x_full = np.array([[cv2.resize(f, (h, w)) for f in seq] for seq in x_raw])
    x_cropped = np.array([[f[168:280, 168:280] for f in seq] for seq in x_raw])
    x_last_bigger = np.array([seq[-1] for seq in x_raw])
    
    
    #x_full = np.array([[cv2.resize(f.astype(np.float32) - mean_frame, (h, w)) for f in seq] for seq in input_images_in_seqs])
    #x_cropped = np.array([[(f.astype(np.float32)-mean_frame)[168:280, 168:280] for f in seq] for seq in input_images_in_seqs])
    #x_last_bigger = np.array([cv2.resize(seq[-1], (4*h, 4*w)) for seq in x_full])
    
    #x_last_bigger = np.array([cv2.resize(seq[-1].astype(np.float32) - mean_frame, (4*h, 4*w)) for seq in input_images_in_seqs])
    
    #x_full.shape = (?, 16, 112, 112, 3)
    #x_cropped.shape = (?, 16, 112, 112, 3)
    #x_last_bigger.shape = (?, 448, 448, 3)

    x_full = x_full.astype(np.float32)
    x_cropped = x_cropped.astype(np.float32)
    x_last_bigger = x_last_bigger.astype(np.float32)
    
    y_full = annotations_in_seqs[:, -1, :, :, 0]
    #y_full.shape = (?, 448, 448)
    #normalization
    temp = np.zeros(y_full.shape).astype(np.uint8)
    for i in range(len(y_full)):
        y = y_full[i]
        if y.max() != 0:
            temp[i] = ((y/y.max())*255).astype(np.uint8)
    y_full = temp
    
    y_cropped = y_full[:, 168:280, 168:280]
    #y_full.shape = (?, 448, 448)
    #y_cropped.shape = (?, 112, 112)
    
    y_full = y_full.reshape((-1, 448*448))
    y_cropped = y_cropped.reshape((-1, 112*112))
    #y_full.shape = (?, 200704)
    #y_cropped.shape = (?, 12544)
    
    return x_cropped, x_full, x_last_bigger, y_cropped, y_full


def y2img(y_full):
    #y_full.shape = (?, 200704)
    
    y_norm = np.clip(y_full, a_min=0, a_max=255)
    temp = np.zeros(y_norm.shape).astype(np.uint8)
    for i in range(len(y_norm)):
        y = y_norm[i]
        if y.max() != 0:
            temp[i] = ((y/y.max())*255).astype(np.uint8)
    y_norm = temp
    y_norm = y_norm.reshape((-1, 448, 448))
    y_images = np.array([misc.imresize(y, args.display_size) for y in y_norm])
    y_images = y_images[:,:,:,None]
    
    return y_images
    

validation_losses = []
n_iteration = np.ceil(len(
    validation_dataset_reader.data_point_names)/args.batch_size).astype(np.int)

if args.data_dir != 'data/':
    dir_name = args.model_dir+'prediction_iter_'+args.model_iteration+'_for_'+args.data_dir
else:
    dir_name = args.model_dir+'prediction_iter_'+args.model_iteration+'/'
if not os.path.isdir(dir_name):
    os.makedirs(dir_name)
    
dfs = []

for itr in tqdm(range(n_iteration)):
    batch = validation_dataset_reader.next_batch_in_seqs(\
        args.batch_size, args.n_steps)
    #pdb.set_trace()
    valid_input_images_in_seqs = validation_dataset_reader.get_images_in_seqs(batch)
    valid_annotations_in_seqs = \
        validation_dataset_reader.\
        get_annotations_in_seqs(batch, desired_size = args.gaze_map_size)            
        
    x_cropped, x_full, x_last_bigger, y_cropped, y_full = \
        prepare_data(valid_input_images_in_seqs, valid_annotations_in_seqs, mean_frame)

    #do testing
    #loss = model.test_on_batch([x_cropped, x_full, x_last_bigger], [y_cropped, y_full])[0]
    res = model.predict_on_batch([x_cropped, x_full, x_last_bigger])
    valid_pred_images = y2img(res[1])
    #valid_gazemaps = y2img(valid_annotations_in_seqs.reshape(-1, 448*448))
    #valid_gazemaps = y2img(y_full)
    
    #pdb.set_trace()
    
    resized_pred_images = []
    for i in range(len(valid_pred_images)):
        resized_pred_image = cv2.resize(valid_pred_images[i], (args.output_map_size[1], args.output_map_size[0]))
        fpath = dir_name + batch[i][-1] + '.jpg'
        misc.imsave(fpath, resized_pred_image)
        
    
    
    
        





