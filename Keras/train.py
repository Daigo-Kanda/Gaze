import os
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from load_data import load_data_from_npz, load_batch, load_data_names, load_batch_from_names_random
from Network.network_separate_eye_1 import get_eye_tracker_model
import tensorflow as tf
import pickle

# generator for data loaded from the npz file
def generator_npz(data, batch_size, img_ch, img_cols, img_rows):
    while True:
        for it in list(range(0, data[0].shape[0], batch_size)):
            x, y = load_batch([l[it:it + batch_size] for l in data], img_ch, img_cols, img_rows)
            yield x, y


# generator with random batch load (train)
def generator_train_data(names, path, batch_size, img_ch, img_cols, img_rows):
    while True:
        x, y = load_batch_from_names_random(names, path, batch_size, img_ch, img_cols, img_rows)
        yield x, y


# generator with random batch load (validation)
def generator_val_data(names, path, batch_size, img_ch, img_cols, img_rows):
    while True:
        x, y = load_batch_from_names_random(names, path, batch_size, img_ch, img_cols, img_rows)
        yield x, y


def train(args):
    # todo: manage parameters in main
    if args.data == "big":
        dataset_path = "/home/daigokanda/GazeCapture_DataSet"
    if args.data == "small":
        dataset_path = "/home/daigokanda/GazeCapture_DataSet/eye_tracker_train_and_val.npz"

    if args.data == "big":
        train_path = "/home/daigokanda/GazeCapture_DataSet/smallSet/train"
        val_path = "/home/daigokanda/GazeCapture_DataSet/smallSet/validation"
        test_path = "/home/daigokanda/GazeCapture_DataSet/smallSet/test"

    print("{} dataset: {}".format(args.data, dataset_path))

    # train parameters
    n_epoch = args.max_epoch
    batch_size = args.batch_size
    patience = args.patience

    # image parameter
    img_cols = 224
    img_rows = 224
    img_ch = 3

    # model
    model = get_eye_tracker_model(img_ch, img_cols, img_rows)

    # model summary
    model.summary()

    # models
    # print("Loading models...",  end='')
    # weights_path = "models/models.003-4.05525.hdf5"
    # model.load_weights(weights_path)
    # print("Done.")

    # optimizer
    sgd = SGD(lr=1e-1, decay=5e-4, momentum=9e-1, nesterov=True)
    adam = Adam(lr=1e-3)

    # compile model
    model.compile(optimizer=adam, loss='mse', metrics=['mae'])

    # data
    # todo: parameters not hardocoded
    if args.data == "big":
        # train data
        train_names = load_data_names(train_path)
        # validation data
        val_names = load_data_names(val_path)
        # test data
        test_names = load_data_names(test_path)

    if args.data == "small":
        train_data, val_data = load_data_from_npz(dataset_path)

    # debug
    # x, y = load_batch([l[0:batch_size] for l in train_data], img_ch, img_cols, img_rows)
    # x, y = load_batch_from_names(train_names[0:batch_size], dataset_path, img_ch, img_cols, img_rows)

    # last dataset checks
    if args.data == "small":
        print("train data sources of size: {} {} {} {} {}".format(
            train_data[0].shape[0], train_data[1].shape[0], train_data[2].shape[0],
            train_data[3].shape[0], train_data[4].shape[0]))
        print("validation data sources of size: {} {} {} {} {}".format(
            val_data[0].shape[0], val_data[1].shape[0], val_data[2].shape[0],
            val_data[3].shape[0], val_data[4].shape[0]))

    if args.data == "big":
        model.fit_generator(
            generator=generator_train_data(train_names, dataset_path, batch_size, img_ch, img_cols, img_rows),
            steps_per_epoch=(len(train_names)) / batch_size,
            epochs=n_epoch,
            verbose=1,
            validation_data=generator_val_data(val_names, dataset_path, batch_size, img_ch, img_cols, img_rows),
            validation_steps=(len(val_names)) / batch_size,
            callbacks=[EarlyStopping(patience=patience),
                       ModelCheckpoint("weights_big/models.{epoch:03d}-{val_loss:.5f}.hdf5", save_best_only=True)
                       ]
        )
    if args.data == "small":
        history = model.fit_generator(
            generator=generator_npz(train_data, batch_size, img_ch, img_cols, img_rows),
            steps_per_epoch=(train_data[0].shape[0]) / batch_size,
            epochs=n_epoch,
            verbose=1,
            validation_data=generator_npz(val_data, batch_size, img_ch, img_cols, img_rows),
            validation_steps=(val_data[0].shape[0]) / batch_size,
            callbacks=[EarlyStopping(patience=patience),
                       ModelCheckpoint("./models/model.{epoch:03d}-{val_loss:.5f}.hdf5", save_best_only=False),
                       tf.keras.callbacks.TensorBoard(
                           log_dir='./logs', histogram_freq=1, write_grads=True, write_images=1, embeddings_freq=1)
                       ]
        )

        model.save('./models/my_model.h5')

        with open('trainHistoryDict.pickle', 'wb') as file_pi:
            pickle.dump(history.history, file_pi)
