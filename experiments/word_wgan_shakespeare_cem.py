# import os
# os.environ["THEANO_FLAGS"] = "mode=FAST_COMPILE,device=cpu,floatX=float32"
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


def decoder(latent_dim, hidden_dim, x_k, n_steps):
    # x, output, (n, x_k+2)
    z = T.fmatrix("z")  # (n, latent_dim)
    W_f = T.fmatrix("W_f")  # z, (latent_dim, hidden_dim)
    U_f = T.fmatrix("U_f")  # h, (hidden_dim, hidden_dim)
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

    def sample_params():
        e = 1e-2
        _W_f = np.random.uniform(-e, e, (latent_dim, hidden_dim))
        _U_f = np.random.uniform(-e, e, (hidden_dim, hidden_dim))
        _V_f = np.random.uniform(-e, e, (x_k + 2, hidden_dim))
        _b_f = np.random.uniform(-e, e, (1, hidden_dim))
        _W_i = np.random.uniform(-e, e, (latent_dim, hidden_dim))
        _U_i = np.random.uniform(-e, e, (hidden_dim, hidden_dim))
        _V_i = np.random.uniform(-e, e, (x_k + 2, hidden_dim))
        _b_i = np.random.uniform(-e, e, (1, hidden_dim))
        _W_c = np.random.uniform(-e, e, (latent_dim, hidden_dim))
        _U_c = np.random.uniform(-e, e, (hidden_dim, hidden_dim))
        _V_c = np.random.uniform(-e, e, (x_k + 2, hidden_dim))
        _b_c = np.random.uniform(-e, e, (1, hidden_dim))
        _W_o = np.random.uniform(-e, e, (latent_dim, hidden_dim))
        _U_o = np.random.uniform(-e, e, (hidden_dim, hidden_dim))
        _V_o = np.random.uniform(-e, e, (x_k + 2, hidden_dim))
        _b_o = np.random.uniform(-e, e, (1, hidden_dim))
        _W_x = np.random.uniform(-e, e, (hidden_dim, x_k + 1))
        _b_x = np.random.uniform(-e, e, (1, x_k + 1))
        params = [_W_f, _U_f, _V_f, _b_f,
                  _W_i, _U_i, _V_i, _b_i,
                  _W_c, _U_c, _V_c, _b_c,
                  _W_o, _U_o, _V_o, _b_o,
                  _W_x, _b_x
                  ]
        return [p.astype(np.float32) for p in params]

    n = z.shape[0]
    outputs_info = [{'initial': T.zeros((n,), dtype='int32')},
                    {'initial': T.zeros((n, hidden_dim), dtype='float32')}]
    x, h = theano.scan(func, outputs_info=outputs_info, non_sequences=[z] + params, n_steps=n_steps)[0]
    # print("X: {}".format(x))
    x = T.transpose(x, (1, 0))
    h = T.transpose(h, (1, 0, 2))
    return z, params, [x - 1, h], sample_params


class Discriminator(object):
    def __init__(self, x_k, n_steps, hidden_dim):
        self.x_k = x_k
        self.hidden_dim = hidden_dim
        constraint = lambda: ClipConstraint(1e-2)
        self.lstm = LSTM(hidden_dim)
        self.lstm.build((None, n_steps, 1))
        for w in self.lstm.trainable_weights:
            # print("Weight: {}".format(w))
            self.lstm.constraints[w] = constraint()
        self.dense = Dense(1, W_constraint=constraint())
        self.dense.build((None, hidden_dim))
        self.weights = self.lstm.trainable_weights + self.dense.trainable_weights
        self.constraints = self.lstm.constraints.copy()
        self.constraints.update(self.dense.constraints)
        # print("Constraints: {}".format(self.constraints))

    def call(self, x):
        return self.dense.call(self.lstm.call(x))


class WGanCEM(object):
    def __init__(self, path, words, latent_dim, hidden_dim):
        data = load_or_create(path, words, lower=True)
        self.words = data["words"]
        self.characters = data["characters"]
        self.charset = data["charset"]
        self.charmap = data["charmap"]
        self.wordcount = len(self.words)
        self.charcount = len(self.charset)
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.n_steps = 8
        self.word_vectors = np.vstack([self.word_to_vector(word).reshape((1, -1)) for word in self.words])
        self.word_vectors = np.expand_dims(self.word_vectors, 2)
        np.random.shuffle(self.word_vectors)
        x_k = self.charcount
        z, params_g, (x_fake, h), sample_params = decoder(latent_dim, hidden_dim, x_k, self.n_steps)
        self.params_g_t = sample_params()
        self.discriminator = Discriminator(x_k, self.n_steps, hidden_dim)
        # x_real = T.fmatrix()
        # x_real._keras_shape = (None, self.n_steps, 1)
        x_real = Input((self.n_steps, 1))
        y_real = T.mean(self.discriminator.call(x_real), axis=None)
        x_fakef = T.cast(x_fake.dimshuffle((0, 1, 'x')), 'floatX')
        x_fakef._keras_shape = (None, self.n_steps, 1)
        y_fake = T.mean(self.discriminator.call(x_fakef), axis=None)
        loss_d = y_fake - y_real
        self.opt_d = Adam(1e-4)
        updates_d = self.opt_d.get_updates(self.discriminator.weights, self.discriminator.constraints, loss_d)
        self.batch_size = 64
        self.batch_count = 8
        self.unroll_depth = 16
        self.generator_count = 8
        self.train_d_function = K.function([x_real, z] + params_g, [loss_d], updates=updates_d)
        self.predict_d_function = K.function([x_real], [y_real])

        self.predict_g_function = K.function([z] + params_g, [x_fake])
        self.test_g_function = K.function([z] + params_g, [-y_fake])

        self.backup_params = self.discriminator.weights + self.opt_d.weights
        self.backup_values = None

    def discriminator_backup(self):
        self.backup_values = [p.get_value() for p in self.backup_params]

    def discriminator_load(self):
        for p, v in zip(self.backup_params, self.backup_values):
            p.set_value(v)

    def word_to_vector(self, word):
        assert len(word) <= self.n_steps
        ar = [self.charmap[c] + 1 for c in word]
        while len(ar) < self.n_steps:
            ar.append(0)
        vec = np.array(ar)
        return vec

    def vector_to_word(self, vec):
        assert vec.ndim == 1
        str = ""
        for i in range(vec.shape[0]):
            c = vec[i]
            if c > 0:
                str += self.charset[c - 1]
        return str

    def sample_word(self):
        i = np.random.randint(0, self.wordcount)
        return self.words[i]

    def latent_sample(self, n):
        return np.random.normal(0, 1, (n, self.latent_dim))

    def train_d(self):
        losses = []
        for _ in range(self.batch_count):
            losses.append(self.train_d_batch(self.params_g_t))
        return np.average(losses, axis=None)

    def train_d_batch(self, generator_params):
        x_ind = np.random.randint(0, self.word_vectors.shape[0], (self.batch_size,))
        x_real = self.word_vectors[x_ind, :, :]
        z = self.latent_sample(x_real.shape[0])
        return self.train_d_function([x_real, z] + generator_params)[0]

    def test_g(self, params):
        for _ in range(self.unroll_depth):
            self.train_d_batch(params)
        losses = []
        for _ in range(self.batch_count):
            z = self.latent_sample(self.batch_size)
            losses.append(np.mean(self.test_g_function([z] + params), axis=None))
        self.discriminator_load()
        return np.mean(losses, axis=None)

    def parameter_sampling(self):
        eps = 1e-1
        new_params = [param + np.random.normal(0, eps, param.shape) for param in self.params_g_t]
        return new_params

    def train_g(self):
        self.discriminator_backup()
        best_params = self.params_g_t
        best_loss = self.test_g(self.params_g_t)
        original_loss = best_loss
        for _ in range(self.generator_count):
            newparams = self.parameter_sampling()
            newloss = self.test_g(newparams)
            if newloss < best_loss:
                best_params = newparams
                best_loss = newloss
        self.params_g_t = best_params
        tqdm.write("G training: {} -> {}".format(original_loss, best_loss))
        return best_loss

    def train(self, nb_epoch, nb_batch, nb_batch_d, path):
        for epoch in tqdm(range(nb_epoch), desc="Training"):
            d_loss = []
            g_loss = []
            for _ in tqdm(range(nb_batch), desc="Epoch {}".format(epoch)):
                for _ in range(nb_batch_d):
                    d_loss.append(self.train_d_batch(self.params_g_t))
                g_loss.append(self.train_g())
            d_loss = np.mean(d_loss, axis=None)
            g_loss = np.mean(g_loss, axis=None)
            tqdm.write("Epoch: {}, D loss: {}, G loss: {}".format(epoch, d_loss, g_loss))
            self.write_samples(path.format(epoch))

    def write_samples(self, path):
        z = self.latent_sample(self.batch_size)
        preds = self.predict_g_function([z] + self.params_g_t)[0]
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        with open(path, 'w') as f:
            for i in range(self.batch_size):
                word = self.vector_to_word(preds[i, :])
                print("Generated: {}".format(word))
                f.write(word)
                f.write("\n")


def main():
    path = "output/words-shakespeare.pkl"
    outputpath = "output/word-wgan-shakespeare-cem/epoch-{:06d}.txt"
    latent_dim = 100
    hidden_dim = 256
    model = WGanCEM(path, shakespeare.words, latent_dim, hidden_dim)
    model.train(nb_epoch=1000, nb_batch=64, nb_batch_d=8, path=outputpath)


if __name__ == "__main__":
    main()
