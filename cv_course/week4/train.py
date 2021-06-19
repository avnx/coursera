import math
import os
from os.path import join
from copy import copy, deepcopy
from collections import Counter

from keras.models import Model, Sequential, load_model
from keras.layers import Flatten, Dense, Activation, Input, Conv2D, MaxPooling2D, Dropout
from keras.layers import Convolution2D, MaxPooling2D
from keras import backend as K
import numpy as np
import pandas as pd
import cv2
from skimage import color, transform
import h5py
from keras.models import load_model
from sklearn import preprocessing
from keras.callbacks import ModelCheckpoint
from keras.losses import SparseCategoricalCrossentropy
from keras.preprocessing.image import ImageDataGenerator
from skimage.io import imread
import tensorflow as tf
from keras.optimizers import RMSprop
from dlib import get_frontal_face_detector, shape_predictor, rectangle


def get_image(filepath):
    image = cv2.imread(filepath, 1)
    if len(image.shape) == 2: # image is gray
        image = color.gray2rgb(image)
    else:
        image = image[:,:,::-1]
    return image[:, :, :3]


def load_image_data(dir_name = 'Face_Recognition_data/image_classification'):
    """Your implementation"""
    y_train = pd.read_csv(os.path.join(dir_name, 'train/y_train.csv'))
    y_test = pd.read_csv(os.path.join(dir_name, 'test/y_test.csv'))
    X_train, X_test = {}, {}
    for i, row in y_train.iterrows():
        X_train[row['filename']] = get_image(os.path.join(dir_name, 'train/images', row['filename']))
    for i, row in y_test.iterrows():
        X_test[row['filename']] = get_image(os.path.join(dir_name, 'test/images', row['filename']))
    return X_train, y_train.set_index('filename').to_dict()['class_id'], X_test, y_test.set_index('filename').to_dict()['class_id']


def dotproduct(v1, v2):
    return sum((a*b) for a, b in zip(v1, v2))

def length(v):
    return math.sqrt(dotproduct(v, v))

def angle(v1, v2):
    return math.acos(dotproduct(v1, v2) / (length(v1) * length(v2)))


def get_new_coords(x, y, alpha, img_shape):
    center_x, center_y = img_shape[1] // 2, img_shape[0] // 2
    x, y = x - center_x, y - center_y
    new_x = center_x + x * math.cos(alpha) + y * math.sin(alpha)
    new_y = center_y + y * math.cos(alpha) - x * math.sin(alpha)
    return new_x, new_y


def transform_face(image, image_eyes):
    image_center = tuple(np.array(image.shape[:2]) / 2)
    x1, y1 = image_eyes[0]
    x2, y2 = image_eyes[1]
    vector = (x2 - x1, y2 - y1)

    alpha = math.copysign(angle(vector, (1, 0)), vector[1])

    rotated = transform.rotate(image, alpha * 180 / np.pi)

    new_x1, new_y1 = get_new_coords(x1, y1, alpha, image.shape)
    new_x2, new_y2 = get_new_coords(x2, y2, alpha, image.shape)

    eye_distance = new_x2 - new_x1
    y_lower, y_upper = int(new_y1 - 2 * eye_distance), int(new_y1 + 2 * eye_distance)
    x_lower, x_upper = int(new_x1 - eye_distance), int(new_x2 + eye_distance)

    y_lower, x_lower = max(0, y_lower), max(0, x_lower)

    crop_img = transform.resize(rotated[y_lower:y_upper, x_lower:x_upper], [224,224,3])

    return crop_img
    

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


class Classifier():
    def __init__(self, nn_model, y_train):
        """Your implementation"""
        assert type(y_train) == list, 'y_train is not a list'
        n_classes = len(set(y_train))

        for layer in model.layers:
            layer.trainable = False
            if layer.name == 'fc6':
                break
        network_output = model.get_layer('fc6').output
        feature_extraction_model = Model(model.input, network_output)
        x = Dense(n_classes)(feature_extraction_model.output)

        self.model = Model(feature_extraction_model.input, outputs=x)
        
        lr_decayed_fn = tf.keras.optimizers.schedules.ExponentialDecay(0.0015, 500, 0.6)

        self.model.compile(loss=SparseCategoricalCrossentropy(from_logits=True),
              optimizer=RMSprop(),
              metrics=['accuracy'])
        
        self.label_encoder = preprocessing.LabelEncoder()
        self.label_encoder.fit(list(y_train))


    def fit_datagen(self, X_train):
        self.datagen=ImageDataGenerator(horizontal_flip=True,
                           width_shift_range=0.2,
                           height_shift_range=0.2,
                           samplewise_center =  True,
                           samplewise_std_normalization = True)
        self.datagen.fit(X_train)


    def fit(self, X_train, Y_train, X_val, Y_val, epochs):
        """Your implementation"""
        Y_train = self.label_encoder.transform(Y_train)
        Y_val = self.label_encoder.transform(Y_val)
        X_train = np.array(preprocess_imgs(X_train))
        X_val = np.array(preprocess_imgs(X_val))
        self.fit_datagen(X_train)
        X_val = self.datagen.standardize(X_val)
        print('preprocessing done')
        history = self.model.fit_generator(self.datagen.flow(X_train, Y_train, batch_size=BATCH_SIZE),
                                           validation_data=(X_val, Y_val),
                                           epochs=epochs, steps_per_epoch=len(X_train) // BATCH_SIZE,
                                           callbacks=[ModelCheckpoint('checkpoint/{epoch:02d}-{val_loss:.2f}.hdf5', save_best_only=True)]
                                 ) 


    def classify_images(self, X_test):
        """Your implementation"""
        preds = self.model.predict(
                    self.datagen.standardize(
                        preprocess_imgs(X_test)))
        return self.label_encoder.inverse_transform(preds)
        
       
        
    def classify_videos(self, test_video):
        """Your implementation"""
        final_preds = []
        for k, frames in test_video.items():
            preds = self.model.predict(
                        self.datagen.standardize(
                            preprocess_imgs(frames)))
            final_preds.append(np.argmax(np.bincount(preds)))
        return self.label_encoder.inverse_transform(final_preds)


x_train, y_train, x_test, y_test = load_image_data()

face_detector = get_frontal_face_detector()
predictor = shape_predictor('shape_predictor_68_face_landmarks.dat')

model = load_model('face_recognition_model.h5')


BATCH_SIZE = 12

img_classifier = Classifier(model, list(y_train.values()))
print('start fit')
img_classifier.fit(list(x_train.values()),
                   list(y_train.values()),
                   list(x_test.values()),
                   list(y_test.values()),
                   epochs=10
                  )
