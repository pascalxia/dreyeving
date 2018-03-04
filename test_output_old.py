from utils_old import getCoarse2FineModel, predict_folder
from keras.optimizers import Adam
import pdb
from keras.models import Model
from keras.layers import Input
import os
from os.path import join
import sys
import cv2
import numpy as np
from tqdm import tqdm
from collections import deque


if __name__ == '__main__':

    output_dir_root = 'out'
    #output_dir_root = 'out_for_bdd'
    weights_file = 'weights/model_weights.h5'
    dreyeve_data_dir = 'data_sample/54'
    #dreyeve_data_dir = '/data/validation/camera_images'
    
    sample_rate = 3
    frame_rate = 25
    # parameters (no need to edit)
    t, c, w, h = 16, 3, 112, 112
    upsample = 4
    

    # load model for prediction
    model = getCoarse2FineModel(summary=True)
    opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(optimizer=opt,
                  loss={'cropped_output': 'mse', 'full_fine_output': 'mse'},
                  loss_weights={'cropped_output': 1.0, 'full_fine_output': 1.0})

    # load pre-trained weights
    model.load_weights(weights_file)
    
    pdb.set_trace()
    
    # make a test model-------------------
    videoclip_cropped = Input((c, t, h, w), name='test_input1')
    test_layer = model.get_layer('sequential_1').get_layer('conv1')
    
    test_model = Model(input=videoclip_cropped, output=test_layer(videoclip_cropped))


    folder_in = dreyeve_data_dir
    output_path = output_dir_root
    mean_frame_path = 'data_sample/dreyeve_mean_frame.png'
    
    
    
    # load frames to predict
    frames = []
    frame_list = os.listdir(folder_in)
    
    frame_list.sort()
    frame_list = frame_list[:t+1]
    
    i = 16
    
    # prepare input------------------
    mean_frame = cv2.imread(mean_frame_path)
    frames = []
    for frame_name in frame_list[i-t:i]:
        frame = cv2.imread(join(folder_in, frame_name))
        frames.append(frame.astype(np.float32) - mean_frame)
    
    sys.stdout.write('\r{0}: predicting on frame {1}...'.format(folder_in, frame_list[i]))

    # convert to array
    x = np.array(frames)

    x_last_bigger = cv2.resize(x[-1, :, :, :], (h*upsample,w*upsample))
    x_last_bigger = x_last_bigger.transpose(2, 0, 1)
    x_last_bigger = x_last_bigger[None, :]

    x = np.array([cv2.resize(f, (h, w)) for f in x])
    x = x[None, :]
    x = x.transpose(0, 4, 1, 2, 3).astype(np.float32)
    
    # predict for a single batch------------------------
    pdb.set_trace()
    res = test_model.predict_on_batch(x)
    print('to be tested')
    

