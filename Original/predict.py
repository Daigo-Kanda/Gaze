# predict particular image
import numpy as np
import tensorflow as tf

import Original.ITrackerData_person_tensor as data_gen

itracker_data = data_gen.ITrackData()

data = itracker_data.getData(batch_size=1, memory_size=150, dataset_path='/mnt/data/DataSet/GazeCapture_pre/00252/')
data_list = itracker_data.getDataList()

model = tf.keras.models.load_model('models/models.046-2.46558.hdf5')

meta_file = '/mnt/data/DataSet/GazeCapture_pre/00252/metadata_person.mat'

for mini_data, face_img_path, gaze_point, grid in zip(data[0], data_list[2], data_list[4], data_list[3]):

    if face_img_path[-9:-4] == '00757':
        np.savetxt('grid.txt', grid)
        history = model.predict(mini_data)
        print("id:{}, predict:{}, true:{}, grid:{}".format(face_img_path[-9:-4], history, gaze_point, grid))
        # print(gaze_point)
