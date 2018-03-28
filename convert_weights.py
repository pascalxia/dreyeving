import h5py
import numpy as np
from keras.utils.conv_utils import convert_kernel


# this is the old weight file for Keras 1.0.3 and Theano back-end
weights_file_old = 'weights/model_weights.h5'
f_old = h5py.File(weights_file_old, 'r')

# this weight file is generated when the network is trained with Keras 2.0.6 and Tensorflow back-end
# the weight values in this file are going to be overwritten to the values converted from the old weight file
weights_file_new = 'weights/model_weights_tf.h5'
f_new = h5py.File(weights_file_new, 'r+')



def send_weights(name, obj):
    if isinstance(obj, h5py.Dataset):
        # change weight names
        if name.startswith('convolution2d'):
            name = name.replace('convolution2d', 'conv2d')
            name = name.replace('_W', '/kernel:0')
            name = name.replace('_b', '/bias:0')
        elif name.startswith('sequential_1/batchnormalization'):
            name = name.replace('batchnormalization', 'batch_normalization')
            name = name.replace('_beta', '/beta:0')
            name = name.replace('_gamma', '/gamma:0')
            name = name.replace('_running_mean', '/moving_mean:0')
            name = name.replace('_running_std', '/moving_variance:0')
        elif name.startswith('sequential_1/conv'):
            name = name.replace('convolution2d', 'conv2d')
            name = name.replace('_W', '/kernel:0')
            name = name.replace('_b', '/bias:0')
        
        # read weight values
        weights = obj[...]
        
        # convert convolution kernels
        if weights.ndim==5:
            weights = weights.transpose([2,3,4,1,0])
            indices = [slice(None, None, -1)]*3 + [slice(None, None)]*2
            weights = weights[indices]
        elif weights.ndim==4:
            weights = weights.transpose([2,3,1,0])
            indices = [slice(None, None, -1)]*2 + [slice(None, None)]*2
            weights = weights[indices]
            
        # convert std to variance for batch normalization
        if name.endswith('/moving_variance:0'):
            weights = np.square(weights)
        
        # write the converted weights values into the new file
        f_new[name][...] = weights
        
        

f_old.visititems(send_weights)


