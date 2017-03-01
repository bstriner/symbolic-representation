import os

# os.environ["THEANO_FLAGS"] = "mode=FAST_COMPILE,device=cpu,floatX=float32"
import theano
import theano.tensor as T
from keras.initializations import zero, glorot_uniform
from keras.layers import Input
import numpy as np
from keras.optimizers import Adam, RMSprop
from gym_symbolic_representation.constraints import ClipConstraint
from gym_symbolic_representation.datasets import shakespeare
from gym_symbolic_representation.datasets import short
from gym_symbolic_representation.datasets.processing import load_or_create
from tqdm import tqdm
from theano.tensor.shared_randomstreams import RandomStreams
from theano import function


def discriminator_function(x, h, y,
                           W_h, U_h, b_h,
                           W_f, b_f,
                           W_i, b_i,
                           W_c, b_c,
                           W_o, b_o,
                           W_j, b_j,
                           W_v, b_v):
    """
    h = hidden state (n, hidden_dim) [prior]
    y = last discriminator value (unused) [prior]
    x = output (n,) int [sequence]
    """
    # print("discriminator_function h: {}, {}".format(h, h.ndim))
    # print("discriminator_function y: {}, {}".format(y, y.ndim))
    hh = T.tanh(T.dot(h, W_h) + U_h[x, :] + b_h)
    #hh = T.dot(h, W_h) + U_h[x, :] + b_h
    f = T.nnet.sigmoid(T.dot(hh, W_f) + b_f)
    i = T.nnet.sigmoid(T.dot(hh, W_i) + b_i)
    o = T.nnet.sigmoid(T.dot(hh, W_c) + b_c)
    w = T.tanh(T.dot(hh, W_o) + b_o)
    h_t = f * h + i * w
    h1 = o * h_t
    h2 = T.tanh(T.dot(h1, W_j) + b_j)
    y_t = T.dot(h2, W_v) + b_v
    y_t = y_t[:, 0]
    # print("discriminator_function h_t: {}, {}".format(h_t, h_t.ndim))
    # print("discriminator_function y_t: {}, {}".format(y_t, y_t.ndim))
    return h_t, y_t


# sequences (if any), prior result(s) (if needed), non-sequences (if any)
def policy_function(e, ex, h, x, z, exploration_probability,
                    W_h, U_h, V_h, b_h,
                    W_f, b_f,
                    W_i, b_i,
                    W_c, b_c,
                    W_o, b_o,
                    W_j, b_j,
                    W_v, b_v):
    """
    e = random [0-1] [sequence]
    ex = random [0-k] [sequence]
    h = hidden state (n, hidden_dim) [prior]
    x = last output (n,) int [prior]
    z = latent vector (n, latent_dim) [non-sequence]
    """
    # print("policy_function h: {}, {}".format(h, h.ndim))
    # print("policy_function x: {}, {}".format(x, x.ndim))
    hh = T.tanh(T.dot(z, W_h) + T.dot(h, U_h) + V_h[x, :] + b_h)
    f = T.nnet.sigmoid(T.dot(hh, W_f) + b_f)
    i = T.nnet.sigmoid(T.dot(hh, W_i) + b_i)
    o = T.nnet.sigmoid(T.dot(hh, W_c) + b_c)
    w = T.tanh(T.dot(hh, W_o) + b_o)
    h_t = f * h + i * w
    h1 = o * h_t
    h2 = T.tanh(T.dot(h1, W_j) + b_j)
    v = T.dot(h2, W_v) + b_v
    x_t = T.argmax(v, axis=-1)
    switch = T.gt(e, exploration_probability)
    x_tt = ((switch * x_t) + ((1 - switch) * ex)) + 1
    x_ttt = T.cast(x_tt, "int32")
    # print("policy_function h_t: {}, {}".format(h_t, h_t.ndim))
    # print("policy_function x_ttt: {}, {}".format(x_ttt, x_ttt.ndim))
    return h_t, x_ttt


def value_function(x, a, h, v, z,
                   W_h, U_h, V_h, b_h,
                   W_f, b_f,
                   W_i, b_i,
                   W_c, b_c,
                   W_o, b_o,
                   W_j, b_j,
                   W_v, b_v):
    """
    x = last input (n, 1) int sequence
    a = actions (n, 1) int sequence
    h = hidden state (n, hidden_dim) [prior]
    v = last value (unused) [prior]
    z = latent vector (n, latent_dim) non-sequence
    """
    hh = T.tanh(T.dot(z, W_h) + T.dot(h, U_h) + V_h[x, :] + b_h)
    #hh = T.dot(z, W_h) + T.dot(h, U_h) + V_h[x, :] + b_h
    f = T.nnet.sigmoid(T.dot(hh, W_f) + b_f)
    i = T.nnet.sigmoid(T.dot(hh, W_i) + b_i)
    o = T.nnet.sigmoid(T.dot(hh, W_c) + b_c)
    w = T.tanh(T.dot(hh, W_o) + b_o)
    h_t = f * h + i * w
    h1 = o * h_t
    h2 = T.tanh(T.dot(h1, W_j) + b_j)
    vh = T.dot(h2, W_v) + b_v
    vt = vh[T.arange(vh.shape[0]), a]
    return h_t, vt


# seq, prior result, non sequence
def reward_function(r, vt1, decay):
    return r + vt1 * decay


class Generator(object):
    def __init__(self, name, latent_dim, depth, k, hidden_dim, exploration_probability, exploration_decay_rate):
        """
        z = input (n, latent_dim)
        o = hidden representation (n, depth, hidden_dim)
        x = output (n,depth) (int)
        h = hidden input representation
        z*W
        o*U
        x*V
        """
        self.latent_dim = latent_dim
        self.depth = depth
        self.k = k
        self.hidden_dim = hidden_dim
        # z = T.fmatrix("z")  # input latent samples (n, latent_dim)
        self.exploration_probability = theano.shared(np.float32(exploration_probability),
                                                     "{}_exploration_probability".format(name))
        self.exploration_decay_rate = np.float32(exploration_decay_rate)

        # Hidden representation
        self.W_h = glorot_uniform((latent_dim, hidden_dim), "{}_W_h".format(name))  # z, (latent_dim, hidden_dim)
        self.U_h = glorot_uniform((hidden_dim, hidden_dim), "{}_U_h".format(name))  # h, (hidden_dim, hidden_dim)
        self.V_h = glorot_uniform((k + 2, hidden_dim), "{}_V_h".format(name))  # x, (x_k+2, hidden_dim)
        self.b_h = zero((hidden_dim,), "{}_b_h".format(name))  # (hidden_dim,)

        # Forget gate
        self.W_f = glorot_uniform((hidden_dim, hidden_dim), "{}_W_f".format(name))  # z, (latent_dim, hidden_dim)
        self.b_f = zero((hidden_dim,), "{}_b_f".format(name))  # (hidden_dim,)
        # Input gate
        self.W_i = glorot_uniform((hidden_dim, hidden_dim), "{}_W_i".format(name))  # z, (latent_dim, hidden_dim)
        self.b_i = zero((hidden_dim,), "{}_b_i".format(name))  # (hidden_dim,)
        # Write gate
        self.W_w = glorot_uniform((hidden_dim, hidden_dim), "{}_W_w".format(name))  # z, (latent_dim, hidden_dim)
        self.b_w = zero((hidden_dim,), "{}_b_w".format(name))  # (hidden_dim,)
        # Output
        self.W_o = glorot_uniform((hidden_dim, hidden_dim), "{}_W_i".format(name))  # z, (latent_dim, hidden_dim)
        self.b_o = zero((hidden_dim,), "{}_b_i".format(name))  # (hidden_dim,)
        # Hidden state
        self.W_j = glorot_uniform((hidden_dim, hidden_dim), "{}_W_j".format(name))  # z, (latent_dim, hidden_dim)
        self.b_j = zero((hidden_dim,), "{}_b_j".format(name))  # (hidden_dim,)
        # Value predictions
        self.W_v = glorot_uniform((hidden_dim, k + 1), "{}_W_v".format(name))  # z, (latent_dim, hidden_dim)
        self.b_v = zero((k + 1,), "{}_b_v".format(name))  # (hidden_dim,)
        self.params = [self.W_h, self.U_h, self.V_h, self.b_h,
                       self.W_f, self.b_f,
                       self.W_i, self.b_i,
                       self.W_w, self.b_w,
                       self.W_o, self.b_o,
                       self.W_j, self.b_j,
                       self.W_v, self.b_v]

    def exploration_decay(self):
        new_value = self.exploration_probability.get_value() * self.exploration_decay_rate
        new_value = np.float32(new_value)
        self.exploration_probability.set_value(new_value)

    def policy(self, z, e, ex):
        """

        :param z: input latent sample (n, latent_dim)
        :param e: exploration epsilon [0-1] (n, depth)
        :param ex: input latent sample (int 0-k+1) (n, depth)
        :return: h (n, depth, hidden_dim), x(n, depth) (int)
        """
        er = T.transpose(e, (1, 0))
        exr = T.transpose(ex, (1, 0))
        n = z.shape[0]
        outputs_info = [T.zeros((n, self.hidden_dim), dtype='float32'),
                        T.zeros((n,), dtype='int32')]
        (h, x), _ = theano.scan(policy_function, sequences=[er, exr], outputs_info=outputs_info,
                                non_sequences=[z, self.exploration_probability] + self.params)
        h = T.transpose(h, (1, 0, 2))
        x = T.transpose(x, (1, 0))
        x -= 1
        return h, x

    def value(self, z, x):
        """

        :param z: input latent sample (n, latent_dim)
        :param x: characters (n, depth)
        :return:
        """
        n = z.shape[0]
        xshifted = T.concatenate((T.zeros((n, 1), dtype='int32'), 1 + x[:, :-1]), axis=-1)
        xr = T.transpose(x, (1, 0))
        xshiftedr = T.transpose(xshifted, (1, 0))
        outputs_info = [T.zeros((n, self.hidden_dim), dtype='float32'),
                        T.zeros((n,), dtype='float32')]
        (h, v), _ = theano.scan(value_function, sequences=[xshiftedr, xr], outputs_info=outputs_info,
                                non_sequences=[z] + self.params)
        h = T.transpose(h, (1, 0, 2))
        v = T.transpose(v, (1, 0))
        return h, v


class Discriminator(object):
    def __init__(self, name, depth, k, hidden_dim):
        self.depth = depth
        self.k = k
        self.hidden_dim = hidden_dim

        # Hidden representation
        self.W_h = glorot_uniform((hidden_dim, hidden_dim), "{}_W_h".format(name))  # h, (hidden_dim, hidden_dim)
        self.U_h = glorot_uniform((k + 1, hidden_dim), "{}_U_h".format(name))  # x, (k+1, hidden_dim)
        self.b_h = zero((hidden_dim,), "{}_b_h".format(name))  # (hidden_dim,)

        # Forget gate
        self.W_f = glorot_uniform((hidden_dim, hidden_dim), "{}_W_f".format(name))  # z, (latent_dim, hidden_dim)
        self.b_f = zero((hidden_dim,), "{}_b_f".format(name))  # (hidden_dim,)
        # Input gate
        self.W_i = glorot_uniform((hidden_dim, hidden_dim), "{}_W_i".format(name))  # z, (latent_dim, hidden_dim)
        self.b_i = zero((hidden_dim,), "{}_b_i".format(name))  # (hidden_dim,)
        # Write gate
        self.W_w = glorot_uniform((hidden_dim, hidden_dim), "{}_W_w".format(name))  # z, (latent_dim, hidden_dim)
        self.b_w = zero((hidden_dim,), "{}_b_w".format(name))  # (hidden_dim,)
        # Output
        self.W_o = glorot_uniform((hidden_dim, hidden_dim), "{}_W_i".format(name))  # z, (latent_dim, hidden_dim)
        self.b_o = zero((hidden_dim,), "{}_b_i".format(name))  # (hidden_dim,)
        # Hidden state
        self.W_j = glorot_uniform((hidden_dim, hidden_dim), "{}_W_j".format(name))  # z, (latent_dim, hidden_dim)
        self.b_j = zero((hidden_dim,), "{}_b_j".format(name))  # (hidden_dim,)
        # y predictions
        self.W_y = glorot_uniform((hidden_dim, 1), "{}_W_y".format(name))  # z, (latent_dim, hidden_dim)
        self.b_y = zero((1,), "{}_b_y".format(name))  # (hidden_dim,)
        self.clip_params = [self.W_h, self.U_h, self.W_f, self.W_i, self.W_w, self.W_o, self.W_j, self.W_y]
        self.params = [self.W_h, self.U_h, self.b_h,
                       self.W_f, self.b_f,
                       self.W_i, self.b_i,
                       self.W_w, self.b_w,
                       self.W_o, self.b_o,
                       self.W_j, self.b_j,
                       self.W_y, self.b_y]

    def discriminator(self, x):
        """

        :param x: (n, depth)
        :return: h (n, depth, hidden_dim), y (n, depth)
        """
        n = x.shape[0]
        xr = T.transpose(x, (1, 0))
        outputs_info = [T.zeros((n, self.hidden_dim), dtype='float32'), T.zeros((n,), dtype='float32')]
        (h, y), updates = theano.scan(discriminator_function, outputs_info=outputs_info, sequences=xr,
                                      non_sequences=self.params)
        h = T.transpose(h, (1, 0, 2))
        y = T.transpose(y, (1, 0))
        return h, y


class WGanModel(object):
    def __init__(self, latent_dim, hidden_dim, exploration_probability, clip_value, value_decay, data,
                 batch_size, exploration_decay_rate):
        self.latent_dim = latent_dim
        self.words = data["words"]
        self.depth = 1 + max(len(w) for w in self.words)
        depth = self.depth
        self.hidden_dim = hidden_dim
        self.characters = data["characters"]
        self.charset = data["charset"]
        self.charmap = data["charmap"]
        self.wordcount = len(self.words)
        self.charcount = len(self.charset)
        self.generator = Generator("generator", latent_dim, depth, self.charcount, hidden_dim, exploration_probability,
                                   exploration_decay_rate)
        self.discriminator = Discriminator("discriminator", depth, self.charcount, hidden_dim)
        self.clip_value = np.float32(clip_value)
        self.value_decay = theano.shared(np.float32(value_decay), "value_decay")

        self.batch_size = batch_size
        self.word_vectors = np.vstack([self.word_to_vector(word).reshape((1, -1)) for word in self.words]).astype(
            np.int32)
        xreal = Input((depth,), name="xreal", dtype="int32")
        batch_n = T.iscalar("batch_n")
        srng = RandomStreams(seed=234)
        z = srng.normal(size=(batch_n, latent_dim))
        e = srng.uniform(size=(batch_n, depth), low=0, high=1)
        ex = srng.random_integers(size=(batch_n, latent_dim), low=0, high=self.charcount)
        # z = Input((latent_dim,), name="z", dtype="float32")
        # e = Input((depth,), name="e", dtype="float32")
        # ex = Input((depth,), name="ex", dtype="int32")
        # xreal = T.imatrix("xreal")
        # z = T.fmatrix("z")
        # e = T.fmatrix("e")
        # ex = T.imatrix("ex")
        _, xfake = self.generator.policy(z, e, ex)
        xfake = theano.gradient.zero_grad(xfake)
        # print("xfake: {}, {}".format(xfake, xfake.type))
        # print("xreal: {}, {}".format(xreal, xreal.type))
        _, yfake = self.discriminator.discriminator(xfake)
        _, yreal = self.discriminator.discriminator(xreal)
        dloss = T.mean(yfake, axis=None) - T.mean(yreal, axis=None)
        dconstraints = {p: ClipConstraint(1e-1) for p in self.discriminator.clip_params}
        dopt = Adam(1e-4)
        dupdates = dopt.get_updates(self.discriminator.params, dconstraints, dloss)

        n = z.shape[0]
        outputs_info = [T.zeros((n,), dtype='float32')]
        yfaker = T.transpose(yfake[:, ::-1], (1, 0))
        vtarget, _ = theano.scan(reward_function, outputs_info=outputs_info, sequences=yfaker,
                                 non_sequences=self.value_decay)
        vtarget = T.transpose(vtarget, (1, 0))[:, ::-1]
        # print("vtarget: {}, {}, {}".format(vtarget, vtarget.ndim, vtarget.type))
        _, vpred = self.generator.value(z, xfake)
        gloss = T.mean(T.abs_(vtarget - vpred), axis=None)
        gopt = Adam(1e-5)
        gupdates = gopt.get_updates(self.generator.params, {}, gloss)
        self.discriminator_train_function = theano.function([xreal, batch_n], [dloss], updates=dupdates)
        self.generator_train_function = theano.function([batch_n], [gloss], updates=gupdates)
        self.generator_sample_function = theano.function([batch_n], [xfake])
        self.test_function = theano.function([xreal, batch_n], [dloss, gloss])

    def word_to_vector(self, word):
        assert len(word) <= self.depth
        ar = [self.charmap[c] + 1 for c in word]
        while len(ar) < self.depth:
            ar.append(0)
        vec = np.array(ar).astype(np.int32)
        return vec

    def vector_to_word(self, vec):
        assert vec.ndim == 1
        str = ""
        for i in range(vec.shape[0]):
            c = vec[i]
            if c > 0:
                str += self.charset[c - 1]
        return str

    def matrix_to_words(self, mat):
        n = mat.shape[0]
        return [self.vector_to_word(mat[i, :]) for i in range(n)]

    def sample_word(self):
        i = np.random.randint(0, self.wordcount)
        return self.words[i]

    def latent_sample(self, n):
        return np.random.normal(0, 1, (n, self.latent_dim)).astype(np.float32)

    def real_sample(self, n):
        xind = np.random.randint(0, self.word_vectors.shape[0], (n,))
        return self.word_vectors[xind, :].astype(np.int32)

    def e_sample(self, n):
        return np.random.uniform(0, 1, (n, self.depth)).astype(np.float32)

    def ex_sample(self, n):
        return np.random.randint(0, self.charcount + 1, (n, self.depth)).astype(np.int32)

    def discriminator_train(self):
        xreal = self.real_sample(self.batch_size)
        # z = self.latent_sample(self.batch_size)
        # e = self.e_sample(self.batch_size)
        # ex = self.ex_sample(self.batch_size)
        return self.discriminator_train_function(xreal, self.batch_size)[0]

    def generator_train(self):
        # z = self.latent_sample(self.batch_size)
        # e = self.e_sample(self.batch_size)
        # ex = self.ex_sample(self.batch_size)
        # return self.generator_train_function(z, e, ex)[0]
        return self.generator_train_function(self.batch_size)[0]

    def test(self):
        xreal = self.real_sample(self.batch_size)
        # z = self.latent_sample(self.batch_size)
        # e = self.e_sample(self.batch_size)
        # ex = self.ex_sample(self.batch_size)
        return self.test_function(xreal, self.batch_size)

    def generate_samples(self, n):
        # z = self.latent_sample(n)
        # e = self.e_sample(n)
        # ex = self.ex_sample(n)
        samples = self.generator_sample_function(n)[0]
        return self.matrix_to_words(samples)

    def write_samples(self, path):
        words = self.generate_samples(123)
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        with open(path, 'w') as f:
            for w in words:
                f.write(w)
                f.write("\n")

    def train(self, nb_epoch, nb_batch, nb_batch_discriminator, path):
        for epoch in tqdm(range(nb_epoch), desc="Training"):
            self.write_samples(path.format(epoch))
            dloss = []
            gloss = []
            for _ in tqdm(range(nb_batch), desc="Epoch {}".format(epoch)):
                for _ in range(nb_batch_discriminator):
                    dloss.append(self.discriminator_train())
                gloss.append(self.generator_train())
            dloss = np.mean(dloss, axis=None)
            gloss = np.mean(gloss, axis=None)
            print("Epoch: {}, D loss: {}, G loss: {}".format(epoch, dloss, gloss))
            self.generator.exploration_decay()


def main():
    #path = "output/words-shakespeare-lower.pkl"
    path = "output/words-short-lower.pkl"
    data = load_or_create(path, short.words, lower=True)
    latent_dim = 32
    hidden_dim = 234
    exploration_probability = 0.3
    clip_value = 1e-1
    value_decay = 0.98
    batch_size = 64
    exploration_decay_rate = 0.998
    model = WGanModel(latent_dim, hidden_dim, exploration_probability, clip_value, value_decay, data, batch_size,
                      exploration_decay_rate)
    model.train(10000, 256, 16, "output/wgan-lstm/epoch-{:08d}.txt")


if __name__ == "__main__":
    main()
