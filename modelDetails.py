import tensorflow as tf

model_path = "/mnt/data2/model/Git_Original/Github_model/models/my_model.h5"

model = tf.keras.models.load_model(model_path)

model.summary()

tf.keras.utils.plot_model(model, to_file="model.png", show_shapes=True, expand_nested=True)

