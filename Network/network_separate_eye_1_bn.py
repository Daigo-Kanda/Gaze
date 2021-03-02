import numpy as np
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, MaxPool2D, concatenate, BatchNormalization, \
    Activation
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Model


class ScaledSigmoid(Layer):
    # コンストラクタ
    def __init__(self, alpha, beta, **kwargs):
        self.alpha = alpha
        self.beta = beta
        # 親クラスへの情報の提供
        # **kwargsは任意の数の引数　複数のキーワード引数を辞書として受け取る
        super(ScaledSigmoid, self).__init__(**kwargs)

    def build(self, input_shape):
        super(ScaledSigmoid, self).build(input_shape)

    def call(self, x, mask=None):
        return self.alpha / (1 + np.exp(-x / self.beta))

    def get_output_shape_for(self, input_shape):
        return input_shape


# activation functions
activation = 'relu'
last_activation = 'linear'


# eye model
# img_ch 画像のチャンネル数　入力時はRGBなので3
# img_cols 画像の列
# img_rows 画像の行
def get_eye_model(img_ch, img_cols, img_rows):
    eye_img_input = Input(shape=(img_cols, img_rows, img_ch))

    h = Conv2D(96, (11, 11), activation=activation)(eye_img_input)
    h = MaxPool2D(pool_size=(2, 2))(h)
    h = Conv2D(256, (5, 5), activation=activation)(h)
    h = MaxPool2D(pool_size=(2, 2))(h)
    h = Conv2D(384, (3, 3), activation=activation)(h)
    h = MaxPool2D(pool_size=(2, 2))(h)
    out = Conv2D(64, (1, 1), activation=activation)(h)

    model = Model(inputs=eye_img_input, outputs=out)

    return model


# face model
def get_face_model(img_ch, img_cols, img_rows):
    face_img_input = Input(shape=(img_cols, img_rows, img_ch))

    h = Conv2D(96, (11, 11), activation=activation)(face_img_input)
    h = MaxPool2D(pool_size=(2, 2))(h)
    h = Conv2D(256, (5, 5), activation=activation)(h)
    h = MaxPool2D(pool_size=(2, 2))(h)
    h = Conv2D(384, (3, 3), activation=activation)(h)
    h = MaxPool2D(pool_size=(2, 2))(h)
    out = Conv2D(64, (1, 1), activation=activation)(h)

    model = Model(inputs=face_img_input, outputs=out)

    return model


def get_eye_tracker_model(img_ch, img_cols, img_rows):
    right_eye_input = Input(shape=(img_cols, img_rows, img_ch), name='right_eye_input')
    left_eye_input = Input(shape=(img_cols, img_rows, img_ch), name='left_eye_input')

    h = Conv2D(96, (11, 11), strides=4)(right_eye_input)
    h = BatchNormalization()(h)
    h = Activation('relu')(h)
    h = MaxPool2D(pool_size=3, strides=2)(h)
    h = Conv2D(256, (5, 5), strides=1, padding='same')(h)
    h = BatchNormalization()(h)
    h = Activation('relu')(h)
    h = Conv2D(384, (3, 3), strides=1, padding='same',
               activation=activation)(h)
    right_eye_net = Conv2D(64, (1, 1), strides=1,
                           padding='same', activation=activation)(h)

    h = Conv2D(96, (11, 11), strides=4)(left_eye_input)
    h = BatchNormalization()(h)
    h = Activation('relu')(h)
    h = MaxPool2D(pool_size=3, strides=2)(h)
    h = Conv2D(256, (5, 5), strides=1, padding='same')(h)
    h = BatchNormalization()(h)
    h = Activation('relu')(h)
    h = Conv2D(384, (3, 3), strides=1, padding='same',
               activation=activation)(h)
    left_eye_net = Conv2D(64, (1, 1), strides=1,
                          padding='same', activation=activation)(h)

    # face model
    face_input = Input(shape=(img_cols, img_rows, img_ch), name='face_input')

    h = Conv2D(96, (11, 11), strides=4)(face_input)
    h = BatchNormalization()(h)
    h = Activation('relu')(h)
    h = MaxPool2D(pool_size=3, strides=2)(h)
    h = Conv2D(256, (5, 5), strides=1, padding='same')(h)
    h = BatchNormalization()(h)
    h = Activation('relu')(h)
    h = Conv2D(384, (3, 3), strides=1, padding='same',
               activation=activation)(h)
    face_net = Conv2D(64, (1, 1), strides=1, padding='same',
                      activation=activation)(h)

    # face grid
    face_grid = Input(shape=(25 * 25, 1), name='grid_input')

    # dense layers for eyes
    e = concatenate([left_eye_net, right_eye_net])
    e = Flatten()(e)
    fc_e1 = Dense(128, activation=activation)(e)

    # dense layers for face
    f = Flatten()(face_net)
    fc_f1 = Dense(128, activation=activation)(f)
    fc_f2 = Dense(64, activation=activation)(fc_f1)

    # dense layers for face grid
    fg = Flatten()(face_grid)
    fc_fg1 = Dense(256, activation=activation)(fg)
    fc_fg2 = Dense(128, activation=activation)(fc_fg1)

    # final dense layers
    h = concatenate([fc_e1, fc_f2, fc_fg2])
    fc1 = Dense(128, activation=activation)(h)
    fc2 = Dense(2, activation=last_activation)(fc1)
    # fc2 = Dense(2)(fc1)

    # final model
    final_model = Model(
        inputs=[right_eye_input, left_eye_input, face_input, face_grid],
        outputs=[fc2])

    return final_model
