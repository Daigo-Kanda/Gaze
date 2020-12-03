from Keras.load_data import load_batch,load_data_from_npz

dataset_path = "/home/daigokanda/GazeCapture_DataSet/eye_tracker_train_and_val.npz"


train_data, val_data = load_data_from_npz(dataset_path)
print(train_data[0])

load_batch([l[0:2] for l in train_data], 3, 64, 64)
