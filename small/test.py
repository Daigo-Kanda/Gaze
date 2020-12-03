import os

from load_data import load_data_from_npz, load_batch
import numpy as np
import tensorflow as tf


def test_small():

    dataset_path = "/home/daigokanda/GazeCapture_DataSet/eye_tracker_train_and_val.npz"
    print("Dataset: {}".format(dataset_path))

    weights_path = "/mnt/data2/model/separate/my_model.h5"
    print("Weights: {}".format(weights_path))

    # image parameter
    img_cols = 128
    img_rows = 128
    img_ch = 3

    # test parameter
    batch_size = 64

    model = tf.keras.models.load_model(weights_path)
            
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
    for i, prediction in enumerate(predictions):
        print("PR: {} {}".format(prediction[0], prediction[1]))
        print("GT: {} {} \n".format(y[i][0], y[i][1]))

        err_x.append(abs(prediction[0] - y[i][0]))
        err_y.append(abs(prediction[1] - y[i][1]))

    # mean absolute error
    mae_x = np.mean(err_x)
    mae_y = np.mean(err_y)

    # standard deviation
    std_x = np.std(err_x)
    std_y = np.std(err_y)

    # final results
    print("MAE: {} {} ({} samples)".format(mae_x, mae_y, len(y)))
    print("STD: {} {} ({} samples)".format(std_x, std_y, len(y)))


if __name__ == '__main__':
    test_small()
