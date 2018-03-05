from utils import getCoarse2FineModel, predict_folder
from keras.optimizers import Adam
import pdb


if __name__ == '__main__':

    output_dir_root = 'out_tf/54/'
    #output_dir_root = 'out_for_bdd'
    #weights_file = 'weights/model_weights.h5'
    weights_file = 'logs/run0/weights_iter_0.h5'
    dreyeve_data_dir = 'data_sample/54'
    #dreyeve_data_dir = '/data/validation/camera_images'

    # load model for prediction
    model = getCoarse2FineModel(summary=True)
    opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(optimizer=opt,
                  loss={'cropped_output': 'mse', 'full_fine_output': 'mse'},
                  loss_weights={'cropped_output': 1.0, 'full_fine_output': 1.0})

    # load pre-trained weights
    model.load_weights(weights_file)

    pdb.set_trace()


    # predict on sample data (first 200 frames of run 54 from DR(eye)VE
    predict_folder(model, dreyeve_data_dir,
                  output_path=output_dir_root,
                  mean_frame_path='data_sample/dreyeve_mean_frame.png')
