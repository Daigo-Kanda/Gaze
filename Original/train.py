# osに依存している様々な機能を利用するためのモジュール
import os
import pickle
import time

import ITrackerData as data_gen
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# # os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
# os.environ['TF_CUDNN_USE_AUTOTUNE'] = '0'
# #
# physical_devices = tf.config.list_physical_devices('GPU')
# if len(physical_devices) > 0:
#     for device in physical_devices:
#         tf.config.experimental.set_memory_growth(device, True)
#         print('{} memory growth: {}'.format(device, tf.config.experimental.get_memory_growth(device)))
# else:
#     print("Not enough GPU hardware devices available")
#
# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#     # Restrict TensorFlow to only use the first GPU
#     try:
#         tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
#         logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#         print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
#     except RuntimeError as e:
#         # Visible devices must be set before GPUs have been initialized
#         print(e)

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    # Create 2 virtual GPUs with 5GB memory each
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[1],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5120),
             tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5120)])

        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')

        # tf.config.experimental.set_virtual_device_configuration(
        #     gpus[0],
        #     [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5120),
        #      tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5120)])
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPU,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)

dataset_path = "/home/kanda/GazeCapture_pre"
# dataset_path = "/mnt/data/DataSet/GazeCapture_pre"

# train parameters
n_epoch = 50
batch_size = 256
patience = 10

# image parameter
# 最初の画像のサイズ？
img_cols = 224
img_rows = 224
img_ch = 3

with tf.distribute.MirroredStrategy(devices=logical_gpus,
                                    cross_device_ops=tf.distribute.ReductionToOneDevice(
                                        reduce_to_device="cpu:0")).scope():
    # model
    # model = net.get_eye_tracker_model(img_ch, img_cols, img_rows)
    model = tf.keras.models.load_model("/home/kanda/pycharm/Gaze/Original/models/models.004-10.02091.hdf5")
    # model summary
    model.summary()

    # optimizer
    # sgd = SGD(lr=1e-1, decay=5e-4, momentum=9e-1, nesterov=True)
    # adam = Adam(lr=1e-3)

    # compile model
    # model.compile(optimizer=adam, loss='mse')
    # model.compile(optimizer=adam, loss='mse', metrics=['mae'])

    start = time.time()

    train_gen = data_gen.ITrackerData(dataset_path, 'train', (img_cols, img_rows), (25, 25), batch_size)
    valid_gen = data_gen.ITrackerData(dataset_path, 'val', (img_cols, img_rows), (25, 25), batch_size)

    if not os.path.isdir("./models"):
        os.mkdir("./models")

    if not os.path.isdir("./logs"):
        os.mkdir("./logs")

    # 画像サイズが大きいときはmodel.fitではなくてmodel.fit_generatorを用いる
    history = model.fit(
        x=train_gen,
        steps_per_epoch=len(train_gen),
        initial_epoch=4,
        epochs=n_epoch,
        verbose=1,
        validation_data=valid_gen,
        validation_steps=len(valid_gen),
        # workers=3,
        use_multiprocessing=False,
        callbacks=[EarlyStopping(patience=patience),
                   ModelCheckpoint("models/models.{epoch:03d}-{val_loss:.5f}.hdf5", save_best_only=False,
                                   save_weights_only=False),
                   tf.keras.callbacks.TensorBoard(
                       log_dir='./logs', profile_batch='1,10', histogram_freq=1, write_grads=True, write_images=1,
                       embeddings_freq=1)
                   ]
    )

    model.save('./my_model.h5')

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
