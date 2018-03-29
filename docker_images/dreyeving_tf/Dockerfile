FROM gcr.io/tensorflow/tensorflow:1.3.0-gpu-py3

RUN apt-get -qq -y update && apt-get install -qq -y libsm6 libxext6 libfontconfig1 libxrender1

RUN pip install 'keras==2.0.6' tqdm h5py opencv-python