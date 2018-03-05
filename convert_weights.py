import h5py
import pdb
import numpy as np
from keras.utils.conv_utils import convert_kernel



weights_file = 'weights/model_weights.h5'
f_old = h5py.File(weights_file, 'r')

weights_file = 'logs/run0/weights_iter_0.h5'
#print('logs/run1/weights_iter_1200.h5')
f_new = h5py.File(weights_file, 'r+')


def send_weights(name, obj):
    if isinstance(obj, h5py.Dataset):
        if name.startswith('convolution2d'):
            #pdb.set_trace()
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
        
        weights = obj[...]
        #if weights.ndim>2:
            #pdb.set_trace()
            #weights = convert_kernel(weights)
        #f_new[name][...] = weights.transpose(np.arange(weights.ndim)[::-1])
        
        # convert convolutoin kernels
        if weights.ndim==5:
            weights = weights.transpose([2,3,4,1,0])
            indices = [slice(None, None, -1)]*3 + [slice(None, None)]*2
            weights = weights[indices]
        elif weights.ndim==4:
            weights = weights.transpose([2,3,1,0])
            indices = [slice(None, None, -1)]*2 + [slice(None, None)]*2
            weights = weights[indices]
            
        #convert std to variance for batch normalization
        if name.endswith('/moving_variance:0'):
            weights = np.square(weights)
        
        f_new[name][...] = weights
        
        
          
    
f_old.visititems(send_weights)



txt_file = open('old_weight_shape.txt', 'w')
    
def print_attrs(name, obj):
    #print(name)
    txt_file.write(name)
    txt_file.write('\t')
    if isinstance(obj, h5py.Dataset):
        #print(obj.shape)
        txt_file.write(str(obj.shape))
    txt_file.write('\n')
    
f_old.visititems(print_attrs)
    
f_new.visititems(print_attrs)

txt_file.close()

pdb.set_trace()
