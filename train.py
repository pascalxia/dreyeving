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

#try to reload weights--------------------
if args.model_dir is not None:
    weight_path = args.model_dir + 'weights_iter_' + args.model_iteration + '.h5'
    model.load_weights(weight_path)
    print("Model restored...")

#add conv1*1 at the end
from keras.layers.convolutional import Conv2D
from keras.models import Model
from keras.layers.core import Reshape
from keras.layers import Flatten

full_top = Conv2D(filters=1, kernel_size=1, kernel_initializer='ones', activation='relu', name='full_top')
cropped_top = Conv2D(filters=1, kernel_size=1, kernel_initializer='ones', activation='relu', name='cropped_top')

#pdb.set_trace()
cropped_output = Flatten(name='cropped_top_output')(cropped_top(Reshape((112,112,1))(model.outputs[0])))
full_fine_output = Flatten(name='full_top_output')(full_top(Reshape((448, 448, 1))(model.outputs[1])))
revised_model = Model(input=model.input,
                      output=[cropped_output, full_fine_output])
model = revised_model


opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(optimizer=opt,
              loss={'cropped_top_output': 'mse', 'full_top_output': 'mse'},
              loss_weights={'cropped_top_output': 0.0, 'full_top_output': 1.0})
              
if not os.path.isdir(args.logs_dir):
  os.makedirs(args.logs_dir)
model.save(args.logs_dir+'dreyeving_tf.h5')

    

    

#set up summaries----------
#quick summaries
quick_summaries = []
training_loss = tf.placeholder(tf.float32, shape=(), name='training_loss')
quick_summaries.append(tf.summary.scalar("training_loss", training_loss))
#addtional summaries for the revised model
cropped_top_kernel_plh = tf.placeholder(tf.float32, shape=(), name='cropped_top_kernel')
cropped_top_bias_plh = tf.placeholder(tf.float32, shape=(), name='cropped_top_bias')
full_top_kernel_plh = tf.placeholder(tf.float32, shape=(), name='full_top_kernel')
full_top_bias_plh = tf.placeholder(tf.float32, shape=(), name='full_top_bias')

quick_summaries.append(tf.summary.scalar("cropped_top_kernel", cropped_top_kernel_plh))
quick_summaries.append(tf.summary.scalar("cropped_top_bias", cropped_top_bias_plh))
quick_summaries.append(tf.summary.scalar("full_top_kernel", full_top_kernel_plh))
quick_summaries.append(tf.summary.scalar("full_top_bias", full_top_bias_plh))

quick_summary_op = tf.summary.merge(quick_summaries)


#slow summaries
slow_summaries = []

#input image summary
input_images = tf.placeholder(tf.uint8, 
                             shape=(None, args.display_size[0], args.display_size[1], 3), 
                             name='input_images')

slow_summaries.append(tf.summary.image("training_input_images", 
                                       input_images, max_outputs=2))

#prediction summary
pred_images = tf.placeholder(tf.uint8, 
                             shape=(None, args.display_size[0], args.display_size[1], 1), 
                             name='pred_images')

slow_summaries.append(tf.summary.image("training_pred_images", 
                                       pred_images, max_outputs=2))
slow_summaries.append(tf.summary.histogram('train_pred_images_hist', pred_images))

#ground truth summary
gazemaps = tf.placeholder(tf.uint8, 
                          shape=(None, args.display_size[0], args.display_size[1], 1),
                          name='gazemaps')
slow_summaries.append(tf.summary.image("training_gazemaps", 
                                       gazemaps, max_outputs=2))

slow_summary_op = tf.summary.merge(slow_summaries)


#summaries for validation
valid_summaries = []

validation_loss = tf.placeholder(tf.float32, shape=(), name='validation_loss')
valid_summaries.append(tf.summary.scalar("validation_loss", validation_loss))

valid_summaries.append(tf.summary.image("validation_input_images", 
                                        input_images, max_outputs=2))
valid_summaries.append(tf.summary.image("validation_pred_images", 
                                        pred_images, max_outputs=2))
valid_summaries.append(tf.summary.histogram('valid_pred_images_hist', pred_images))
valid_summaries.append(tf.summary.image("validation_gazemaps", 
                                        gazemaps, max_outputs=2))
valid_summary_op = tf.summary.merge(valid_summaries)



#set up data readers-------------------------------
train_data_points, valid_data_points = \
    data_point_collector.read_datasets_in_sequences(args.data_dir)
    
#pdb.set_trace()
    
train_dataset_reader = \
    BatchDatasetReader.BatchDataset(args.data_dir+'training/',
                         train_data_points, 
                         args.image_size,
                         feature_name=args.feature_name,
                         weight_data=args.weight_data,
                         annotation_threshold=args.annotation_threshold)
validation_dataset_reader = \
    BatchDatasetReader.BatchDataset(args.data_dir+'validation/',
                         valid_data_points, 
                         args.image_size,
                         feature_name=args.feature_name,
                         annotation_threshold=args.annotation_threshold)
                         
                         
#train_dataset_reader = validation_dataset_reader


#set up summary writer
summary_writer = tf.summary.FileWriter(args.logs_dir)

'''
#initialize variables------------------
sess.run(tf.global_variables_initializer())
    

#set up savers------------
saver = tf.train.Saver(max_to_keep=20)
summary_writer = tf.summary.FileWriter(args.logs_dir, sess.graph)


#try to reload weights--------------------
if args.model_dir is not None:
    ckpt = tf.train.get_checkpoint_state(args.model_dir)
    if ckpt and ckpt.model_checkpoint_path:
        vars_stored = [var[0] for var in list_variables(args.model_dir)]
        vars_restore = [v for v in tf.global_variables() if v.name[0:-2] in vars_stored]
        restore_saver = tf.train.Saver(vars_restore)
        restore_saver.restore(sess, ckpt.model_checkpoint_path)
        print("Model restored...")
    else:
        print("Model restore failed...")
'''


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
    

for itr in range(args.max_iteration):
    if not args.weight_data:
        batch = train_dataset_reader.next_batch_in_seqs(args.batch_size, args.n_steps)
    else:
        batch = train_dataset_reader.random_batch_in_seqs(args.batch_size, args.n_steps)
    
    train_input_images_in_seqs = train_dataset_reader.get_images_in_seqs(batch)
    train_annotations_in_seqs = train_dataset_reader.\
        get_annotations_in_seqs(batch,
                                desired_size = args.gaze_map_size)
    
    
    x_cropped, x_full, x_last_bigger, y_cropped, y_full = \
        prepare_data(train_input_images_in_seqs, train_annotations_in_seqs, mean_frame)
    
    #do one step training
    loss = model.train_on_batch([x_cropped, x_full, x_last_bigger], [y_cropped, y_full])[0]

    full_kernel = full_top.get_weights()[0].flatten()[0]
    full_bias = full_top.get_weights()[1][0]
    cropped_kernel = cropped_top.get_weights()[0].flatten()[0]
    cropped_bias = cropped_top.get_weights()[1][0]
    
    
    #do summaries
    if itr % args.quick_summary_period == 0:
        feed_dict = {training_loss: loss,
                     cropped_top_kernel_plh: cropped_kernel,
                     cropped_top_bias_plh: cropped_bias,
                     full_top_kernel_plh: full_kernel,
                     full_top_bias_plh: full_bias}
        summary_str = sess.run(quick_summary_op,
                               feed_dict=feed_dict)
        summary_writer.add_summary(summary_str, itr)
        print("Step: %d, Train_loss:%g" % (itr, loss))
        
    if itr % args.slow_summary_period == 0:
        res = model.predict_on_batch([x_cropped, x_full, x_last_bigger])
        train_pred_images = y2img(res[1])
        #train_gazemaps = y2img(train_annotations_in_seqs.reshape(-1, 448*448))
        train_gazemaps = y2img(y_full)
        
        feed_dict = {
            input_images: [misc.imresize(img, args.display_size) for img in train_input_images_in_seqs.reshape((-1, 448, 448, 3))],
            pred_images: train_pred_images,
            gazemaps: train_gazemaps
        }
        summary_str = sess.run(slow_summary_op, 
                               feed_dict=feed_dict)
        summary_writer.add_summary(summary_str, itr)
    if itr % args.valid_summary_period == 0:
        batch = validation_dataset_reader.random_batch_in_seqs(\
            args.batch_size*args.valid_batch_factor, args.n_steps)
        #pdb.set_trace()
        valid_input_images_in_seqs = validation_dataset_reader.get_images_in_seqs(batch)
        valid_annotations_in_seqs = \
            validation_dataset_reader.\
            get_annotations_in_seqs(batch, desired_size = args.gaze_map_size)            
            
        x_cropped, x_full, x_last_bigger, y_cropped, y_full = \
            prepare_data(valid_input_images_in_seqs, valid_annotations_in_seqs, mean_frame)
    
        #do testing
        loss = model.test_on_batch([x_cropped, x_full, x_last_bigger], [y_cropped, y_full])[0]
        res = model.predict_on_batch([x_cropped, x_full, x_last_bigger])
        valid_pred_images = y2img(res[1])
        #valid_gazemaps = y2img(valid_annotations_in_seqs.reshape(-1, 448*448))
        valid_gazemaps = y2img(y_full)
        
        feed_dict = {
            validation_loss: loss,
            input_images: [cv2.resize(img, args.display_size[::-1]) for img in valid_input_images_in_seqs.reshape((-1, 448, 448, 3))],
            pred_images: valid_pred_images,
            gazemaps: valid_gazemaps
        }
        
        summary_str = sess.run(valid_summary_op, 
                               feed_dict=feed_dict)
                               
        summary_writer.add_summary(summary_str, itr)
        print("%s ---> Validation_loss: %g" % (datetime.datetime.now(), loss))
        
        model.save_weights(args.logs_dir+'weights_iter_%d.h5' % (itr,))
        
    #pdb.set_trace()
    
    
    
        





