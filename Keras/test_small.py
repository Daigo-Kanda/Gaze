import math
import os
import tensorflow as tf
from load_data import load_data_from_npz, load_batch
import numpy as np


def test_small(args):

    dataset_path = "/home/daigokanda/GazeCapture_DataSet/eye_tracker_train_and_val.npz"
    print("Dataset: {}".format(dataset_path))

    model_path = "/mnt/data2/model/Git_Original/Original_model/models/my_model.h5"
    print("model: {}".format(model_path))

    # image parameter
    img_cols = 224
    img_rows = 224
    img_ch = 3

    # test parameter
    batch_size = args.batch_size

    # model
    model = tf.keras.models.load_model(model_path)

    # model summary
    model.summary()

    # data
    train_data, val_data = load_data_from_npz(dataset_path)

    print("Loading testing data...")
    x, y = load_batch([l[:] for l in val_data], img_ch, img_cols, img_rows)
    print("Done.")

    predictions = model.predict(x=x, batch_size=batch_size, verbose=1)

    # print and analyze predictions
    err_x = []
    err_y = []
    err = []
    for i, prediction in enumerate(predictions):
        print("PR: {} {}".format(prediction[0], prediction[1]))
        print("GT: {} {} \n".format(y[i][0], y[i][1]))

        err_x.append(abs(prediction[0] - y[i][0]))
        err_y.append(abs(prediction[1] - y[i][1]))
        err.append(math.sqrt((prediction[0] - y[i][0])**2 + (prediction[1] - y[i][1])**2))

    # mean distance err
    mean_err = np.mean(err)

    # mean absolute error
    mae_x = np.mean(err_x)
    mae_y = np.mean(err_y)

    # standard deviation
    std_x = np.std(err_x)
    std_y = np.std(err_y)

    # final results
    print("MAE: {} {} ({} samples)".format(mae_x, mae_y, len(y)))
    print("STD: {} {} ({} samples)".format(std_x, std_y, len(y)))
    print("DISTANCE_ERR: {} ({} samples)".format(mean_err, len(y)))


if __name__ == '__main__':
    test_small()
