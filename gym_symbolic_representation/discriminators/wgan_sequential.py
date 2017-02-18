from keras.layers.recurrent import LSTM
from keras.layers import Input, Embedding, merge, Dense, Activation, LeakyReLU
from keras.models import Model
from ..objectives import maximize_sign
from keras.optimizers import RMSprop
from ..constraints import ClipConstraint
from ..initializations import uniform_init
import numpy as np


def wgan_sequential_discriminator(x_k, x_len):
    def constraint():
        return ClipConstraint(1e-1)

    x = Input((x_len,), dtype='int32')
    nch = 256
    init = uniform_init(1e-2)
    h = Embedding(x_k+1, nch, W_constraint=constraint(), init=init)(x)
    lstm = LSTM(nch, init=init)
    h = lstm(h)
    lstm.constraints[lstm.U] = constraint()
    lstm.constraints[lstm.W] = constraint()
    h = Dense(nch, W_constraint=constraint(), init=init)(h)
    h = LeakyReLU(0.2)(h)
    h = Dense(nch, W_constraint=constraint(), init=init)(h)
    h = LeakyReLU(0.2)(h)
    y = Dense(1, W_constraint=constraint(), init=init)(h)

    model = Model([x], [y])
    opt = RMSprop(1e-4)
    model.compile(opt, loss=maximize_sign)
    return model
