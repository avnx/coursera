from transform_face import transform_face

from keras.models import Model, Sequential, load_model
from keras.layers import Flatten, Dense, Activation, Input, Conv2D, MaxPooling2D, Dropout
from keras.layers import Convolution2D, MaxPooling2D
from keras import backend as K

import numpy as np
import pandas as pd
import pickle
import cv2
from skimage import color, transform
import os
from copy import copy
from collections import Counter

from keras.preprocessing.image import ImageDataGenerator
from sklearn import preprocessing
import h5py
from keras.models import load_model
from dlib import get_frontal_face_detector, shape_predictor, rectangle


face_detector = get_frontal_face_detector()
predictor = shape_predictor('shape_predictor_68_face_landmarks.dat')


def get_image(filepath):
    image = cv2.imread(filepath, 1)
    if len(image.shape) == 2: # image is gray
        image = color.gray2rgb(image)
    else:
        image = image[:,:,::-1]
    return image[:, :, :3]


def load_video_data(dir_name = 'Face_Recognition_data/video_classification'):
    """Your implementation"""
    y_train = pd.read_csv(os.path.join(dir_name, 'train/y_train.csv'))
    y_test = pd.read_csv(os.path.join(dir_name, 'test/y_test.csv'))
    X_train, X_test = {}, {}
    for i, row in y_train.iterrows():
        X_train[row['filename']] = get_image(os.path.join(dir_name, 'train/images', row['filename']))
    for i, row in y_test.iterrows():
        video_frames = []
        for framename in os.listdir(os.path.join(dir_name, 'test/videos', str(row['filename']))):
            video_frames.append(get_image(os.path.join(dir_name, 'test/videos', str(row['filename']), str(framename))))
        X_test[row['filename']] = video_frames
    return X_train, y_train.set_index('filename').to_dict()['class_id'], X_test, y_test.set_index('filename').to_dict()['class_id']


def preprocess_imgs(imgs):
    """Your implementation"""
    transformed_imgs = []
    
    for img in imgs:
        rect = face_detector(img,1)
        rect.append(rectangle(0,0,img.shape[0],img.shape[1])) #adding a random rectangle,in case none detected
        points = predictor(img, rect[0])
        left, right, top, bottom = max(0, rect[0].left() -20), max(0, rect[0].right()+20), max(0, rect[0].top()-50), max(0, rect[0].bottom()+20)
        face_img = img[top:bottom, left:right]


        eyes = []
        for i in [37, 44]:
            new_x, new_y = points.part(i).x - left, points.part(i).y - top
            eyes.append((new_x, new_y))

        transformed_imgs.append(transform_face(face_img, eyes))
        
    return transformed_imgs


def precalculate_test_samples(filename, frames):
    if os.path.exists('preprocessed_test_video/' + str(filename) + '.pkl'):
        return
    x = preprocess_imgs(frames)
    x = datagen.standardize(x)
    preds = feature_extraction_model.predict(x)
    with open('preprocessed_test_video/' + str(filename) + '.pkl', 'wb') as f:
        pickle.dump(preds, f)
        


video_train, train_labels, video_test, test_labels = load_video_data()
model = load_model('face_recognition_model.h5')
network_output = model.get_layer('fc6').output
feature_extraction_model = Model(model.input, network_output)

label_encoder = preprocessing.LabelEncoder()
label_encoder.fit(list(train_labels.values()))


X_train = np.array(preprocess_imgs(list(video_train.values())))
datagen=ImageDataGenerator(horizontal_flip=True,
                           width_shift_range=0.2,
                           height_shift_range=0.2,
                           samplewise_center =  True,
                           samplewise_std_normalization = True)
datagen.fit(X_train)
print('everything loaded')

i = 0
for j, frames in video_test.items():
    precalculate_test_samples(j, frames)
    print(i)
    i += 1    
