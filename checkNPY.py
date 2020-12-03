import numpy as np
import cv2
import os
import glob
from os.path import join
import json
from data_utility import image_normalization
import tensorflow as tf
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.callbacks import  EarlyStopping, ModelCheckpoint
from load_data import load_data_from_npz, load_data_names, load_batch_from_names_random
from models import get_eye_tracker_model


dataset_path = "/home/daigokanda/itracking/GazeCapture/eye_tracker_train_and_val.npz"

train_data, val_data = load_data_from_npz(dataset_path)

print("train data sources of size: {} {} {} {} {}".format(
    train_data[0].shape[0], train_data[1].shape[0], train_data[2].shape[0],
    train_data[3].shape[0], train_data[4].shape[0]))
print("validation data sources of size: {} {} {} {} {}".format(
    val_data[0].shape[0], val_data[1].shape[0], val_data[2].shape[0],
    val_data[3].shape[0], val_data[4].shape[0]))

print(train_data[0].ndim)
print(train_data[0].shape)
print(train_data[0].size)


print(train_data[3][0].ndim)
print(train_data[3][0].shape)
print(train_data[3][0].size)

print(train_data[3][0].reshape((1, train_data[3][0].shape[0], train_data[3][0].shape[1])).ndim)
print(train_data[3][0].reshape((1, train_data[3][0].shape[0], train_data[3][0].shape[1])).shape)
print(train_data[3][0].reshape((1, train_data[3][0].shape[0], train_data[3][0].shape[1])).size)

# load a batch with data loaded from the npz file
def load_batch(data, img_ch, img_cols, img_rows):

    # create batch structures
    left_eye_batch = np.zeros(shape=(data[0].shape[0], img_ch, img_cols, img_rows), dtype=np.float32)
    right_eye_batch = np.zeros(shape=(data[0].shape[0], img_ch, img_cols, img_rows), dtype=np.float32)
    face_batch = np.zeros(shape=(data[0].shape[0], img_ch, img_cols, img_rows), dtype=np.float32)
    face_grid_batch = np.zeros(shape=(data[0].shape[0], 1, 25, 25), dtype=np.float32)
    y_batch = np.zeros((data[0].shape[0], 2), dtype=np.float32)

    # create batch structures
    left_eye_batch = np.zeros(shape=(data[0].shape[0], img_cols, img_rows, img_ch), dtype=np.float32)
    right_eye_batch = np.zeros(shape=(data[0].shape[0], img_cols, img_rows, img_ch), dtype=np.float32)
    face_batch = np.zeros(shape=(data[0].shape[0], img_cols, img_rows, img_ch), dtype=np.float32)
    face_grid_batch = np.zeros(shape=(data[0].shape[0], 25, 25, 1), dtype=np.float32)
    y_batch = np.zeros((data[0].shape[0], 2), dtype=np.float32)

    # load left eye
    #enumerate インデックスと要素を取得
    for i, img in enumerate(data[0]):
        img = cv2.resize(img, (img_cols, img_rows))
        if save_images:
            cv2.imwrite(join(img_dir, "left" + str(i) + ".png"), img)
        img = image_normalization(img)
        left_eye_batch[i] = img

    # load right eye
    for i, img in enumerate(data[1]):
        img = cv2.resize(img, (img_cols, img_rows))
        if save_images:
            cv2.imwrite("images/right" + str(i) + ".png", img)
        img = image_normalization(img)
        right_eye_batch[i] = img

    # load faces
    for i, img in enumerate(data[2]):
        img = cv2.resize(img, (img_cols, img_rows))
        if save_images:
            cv2.imwrite("images/face" + str(i) + ".png", img)
        img = image_normalization(img)
        face_batch[i] = img

    # load grid faces
    for i, img in enumerate(data[3]):
        if save_images:
            cv2.imwrite("images/grid" + str(i) + ".png", img)
        face_grid_batch[i] = img.reshape((img.shape[0], img.shape[1], 1))

    # load labels
    for i, labels in enumerate(data[4]):
        y_batch[i] = labels

    return [right_eye_batch, left_eye_batch, face_batch, face_grid_batch], y_batch