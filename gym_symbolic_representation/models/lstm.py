import theano
import theano.tensor as T
from gym_symbolic_representation.datasets.processing import load_or_create
from gym_symbolic_representation.datasets import shakespeare
import numpy as np
from keras.layers.recurrent import LSTM
from keras.layers import Dense, Input
from keras.optimizers import Adam
import keras.backend as K
from gym_symbolic_representation.constraints import ClipConstraint
import os
from tqdm import tqdm

class LSTMUnit(object):

    def __init__(self, latent_dim, hidden_dim, x_k, n_steps):
        # x, output, (n, x_k+2)
        self.latent_dim=latent_dim
        self.hidden_dim=hidden_dim
        self.x_k=x_k
        self.n_steps=n_steps
        # z = input, (batch_size, latent_dim)
        self.W_f = T.fmatrix("W_f")  # z, (latent_dim, hidden_dim)
        self.U_f = T.fmatrix("U_f")  # h, (hidden_dim, hidden_dim)
        V_f = T.fmatrix("V_f")  # x, (x_k+2, hidden_dim)
        b_f = T.frow("b_f")  # (hidden_dim,)
        W_i = T.fmatrix()
        U_i = T.fmatrix()
        V_i = T.fmatrix()
        b_i = T.frow()
        W_c = T.fmatrix()
        U_c = T.fmatrix()
        V_c = T.fmatrix()
        b_c = T.frow()
        W_o = T.fmatrix()
        U_o = T.fmatrix()
        V_o = T.fmatrix()
        b_o = T.frow()
        W_x = T.fmatrix()  # x_t, (hidden_dim, x_k+1)
        b_x = T.frow()  # (x_k+1,)

        """
        x = last output (int32)
        h = hidden state
        z = input vector
        """

        # sequences, non-sequences
        def func(_x, _h, _z,
                 _W_f, _U_f, _V_f, _b_f,
                 _W_i, _U_i, _V_i, _b_i,
                 _W_c, _U_c, _V_c, _b_c,
                 _W_o, _U_o, _V_o, _b_o,
                 _W_x, _b_x):
            f = T.nnet.sigmoid(_V_f[_x, :] + T.dot(_z, _W_f) + T.dot(_h, _U_f) + _b_f)
            i = T.nnet.sigmoid(_V_i[_x, :] + T.dot(_z, _W_i) + T.dot(_h, _U_i) + _b_i)
            c = T.tanh(_V_c[_x, :] + T.dot(_z, _W_c) + T.dot(_h, _U_c) + _b_c)
            o = T.nnet.sigmoid(_V_o[_x, :] + T.dot(_z, _W_o) + T.dot(_h, _U_o) + _b_o)
            _h_t = (f * _h) + (i * c)
            _o_t = o * T.tanh(c)
            _x_t = T.argmax(T.dot(_o_t, _W_x) + _b_x, axis=-1) + 1
            switch = T.eq(_x, 1)
            _x_t = switch + ((1 - switch) * _x_t)
            _x_t = T.cast(_x_t, 'int32')
            return _x_t, _h_t

        params = [W_f, U_f, V_f, b_f,
                  W_i, U_i, V_i, b_i,
                  W_c, U_c, V_c, b_c,
                  W_o, U_o, V_o, b_o,
                  W_x, b_x]