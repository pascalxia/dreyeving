from utils import getCoarse2FineModel
from keras.optimizers import Adam
import pdb
import pickle


#set up readout net-----------------
model = getCoarse2FineModel(summary=True)
opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(optimizer=opt,
              loss={'cropped_output': 'mse', 'full_fine_output': 'mse'},
              loss_weights={'cropped_output': 1.0, 'full_fine_output': 1.0})


pdb.set_trace()

model.load_weights('logs/run1/'+'weights_iter_%d.h5' % (3150,))



with open ('outfile', 'rb') as fp:
    itemlist = pickle.load(fp)
