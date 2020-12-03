# osに依存している様々な機能を利用するためのモジュール
import os
import tensorflow as tf
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.callbacks import  EarlyStopping, ModelCheckpoint
from load_data import load_data_from_npz, load_batch, load_data_names, load_batch_from_names_random
# from models import get_eye_tracker_model
from network import models_custom as custom
from network import models as original 
import matplotlib.pyplot as plt
import pickle

import time

# generator for data loaded from the npz file
# オリジナルデータでcnnをするときのみ使用
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

    # GPUを使いやすくするための設定
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.dev

    # todo: manage parameters in main
    # どっちのデータを使用するかの決定
   
    if args.data == "small":
        dataset_path = "/home/daigokanda/itracking/GazeCapture/eye_tracker_train_and_val.npz"

    print("{} dataset: {}".format(args.data, dataset_path))

    # train parameters
    n_epoch = args.max_epoch
    batch_size = args.batch_size
    patience = args.patience

    # image parameter
    # 最初の画像のサイズ？
    img_cols = 64
    img_rows = 64
    img_ch = 3

    # model
    if args.model == "original":
        model = original.get_eye_tracker_model(img_ch, img_cols, img_rows)

    if args.model == "custom":
        model = custom.get_eye_tracker_model(img_ch, img_cols, img_rows)
        
    # model summary
    model.summary()

    # optimizer
    sgd = SGD(lr=1e-1, decay=5e-4, momentum=9e-1, nesterov=True)
    adam = Adam(lr=1e-3)

    # compile model
    # model.compile(optimizer=adam, loss='mse')
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

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

    if args.data == "small":
        start = time.time()

        #画像サイズが大きいときはmodel.fitではなくてmodel.fit_generatorを用いる
        history = model.fit_generator(
            generator=generator_npz(train_data, batch_size, img_ch, img_cols, img_rows),
            steps_per_epoch=(train_data[0].shape[0])/batch_size,
            epochs=n_epoch,
            verbose=1,
            validation_data=generator_npz(val_data, batch_size, img_ch, img_cols, img_rows),
            validation_steps=(val_data[0].shape[0])/batch_size,
            callbacks=[EarlyStopping(patience=patience),
                       ModelCheckpoint("models/models.{epoch:03d}-{val_loss:.5f}.hdf5", save_best_only=True, save_weights_only=False)
                       ]
        )

        model.save('my_model.h5')

        process_time = time.time() - start
        print(process_time)
    
        mae = history.history['mae']
        val_mae = history.history['val_mae']
        loss = history.history['loss']
        val_loss = history.history['val_loss']

        epochs = range(len(mae))

        plt.plot(epochs, mae, 'bo', label='Training acc')
        plt.plot(epochs, val_mae, 'b', label='Validation acc')
        plt.title('Training and validation accuracy')
        plt.legend()

        plt.figure()

        plt.plot(epochs, loss, 'bo', label='Training loss')
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.title('Training and validation loss')
        plt.legend()

        plt.show()

        with open('/trainHistoryDict', 'wb') as file_pi:
            pickle.dump(history.history, file_pi)
