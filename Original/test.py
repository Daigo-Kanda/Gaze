import scipy.io as sio
import tensorflow as tf


def loadMetadata(filename, silent=False):
    try:
        # http://stackoverflow.com/questions/6273634/access-array-contents-from-a-mat-file-loaded-using-scipy-io-loadmat-python
        if not silent:
            print('\tReading metadata from %s...' % filename)
        metadata = sio.loadmat(filename, squeeze_me=True,
                               struct_as_record=False)
    except:
        print('\tFailed to read the meta file "%s"!' % filename)
        return None
    return metadata


meta_data = loadMetadata("mean_face_224.mat")['image_mean']

image = tf.image.decode_jpeg(tf.io.read_file("/mnt/data/DataSet/GazeCapture_pre/00252/appleFace/00757.jpg"), channels=3)

image2 = tf.image.resize(image, [224, 224])

image3 = (image2 / 255.0) - (meta_data / 255.0)

print(image3)
