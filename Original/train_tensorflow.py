# osに依存している様々な機能を利用するためのモジュール
import os
import pickle
import time

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.python.keras.utils.vis_utils import plot_model

from Network import network_separate_eye_1_bn as net
from Original import ITrackerData_tensorflow as data_gen

# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#   # Restrict TensorFlow to only use the first GPU
#   try:
#     tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
#     logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#     print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
#   except RuntimeError as e:
#     # Visible devices must be set before GPUs have been initialized
#     print(e)

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        # tf.config.experimental.set_virtual_device_configuration(
        #     gpus[0],
        #     [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5000),
        #      tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5000)])

        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)

dataset_path = "/kanda_tmp/GazeCapture_pre"
# dataset_path = "/mnt/data/DataSet/GazeCapture_pre"

# train parameters
n_epoch = 50
batch_size = 256
patience = 10
memory_size = 150

# image parameter
# 最初の画像のサイズ？
img_cols = 224
img_rows = 224
img_ch = 3

print("start")
# resource.setrlimit(resource.RLIMIT_AS, (int(20 * 1024 ** 3 * 0.8), 20 * 1024 ** 3))

# with tf.distribute.MirroredStrategy(devices=logical_gpus,
#                                     cross_device_ops=tf.distribute.ReductionToOneDevice(
#                                         reduce_to_device="cpu:0")).scope():


# model
model = net.get_eye_tracker_model(img_ch, img_cols, img_rows)
# model = tf.keras.models.load_model("/kanda_tmp/Gaze/Original/models/models.026-3.02969.hdf5")
# model summary
model.summary()

plot_model(model, to_file='model.png', show_shapes=True)

# optimizer
sgd = SGD(lr=1e-1, decay=5e-4, momentum=9e-1, nesterov=True)
adam = Adam(lr=1e-3)

# compile model
# model.compile(optimizer=adam, loss='mse', metrics=['mae'])

start = time.time()

data = data_gen.getData(batch_size, memory_size, dataset_path)

train_ds = data[0]
val_ds = data[1]

if not os.path.isdir("./models"):
    os.mkdir("./models_tmp")

if not os.path.isdir("./logs"):
    os.mkdir("./logs_tmp")

# 画像サイズが大きいときはmodel.fitではなくてmodel.fit_generatorを用いる
history = model.fit(
    x=train_ds,
    initial_epoch=26,
    epochs=n_epoch,
    verbose=1,
    validation_data=val_ds,
    callbacks=[  # EarlyStopping(patience=patience),
        ModelCheckpoint("models/models.{epoch:03d}-{val_loss:.5f}.hdf5", save_best_only=False,
                        save_weights_only=False),
        tf.keras.callbacks.TensorBoard(
            log_dir='./logs',
            profile_batch='100,150',
            histogram_freq=1, write_grads=True, write_images=1,
            embeddings_freq=1)
    ]
)

model.save('./my_model.hdf5')

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
