from gym import Env
from gym.utils import seeding
from gym import spaces
import numpy as np
from ..datasets import combined
from ..datasets.processing import load_or_create
from ..priors.sequence import sequence
from ..discriminators.wgan_sequential import wgan_sequential_discriminator
from keras.utils.np_utils import to_categorical
import logging


class WordWganEnv(Env):
    """

    Actions:
        0: move left
        1: move right
        2: done
        3+: output characters

    """

    def __init__(self):
        path = "output/words.pkl"
        data = load_or_create(path, combined.words)
        self.words = data["words"]
        self.characters = data["characters"]
        self.charset = data["charset"]
        self.charmap = data["charmap"]
        self.wordcount = len(self.words)
        self.max_steps = 150
        self.max_len = 50
        self.pointer = 0
        self.sequence = sequence()
        self.output = []
        self.action_space = spaces.Discrete(3 + len(self.charset))
        bounds = float('inf')
        self.viewer = None
        self.outputs = []
        self.samples = []
        self.memory_limit = 10000
        self.d_warmup = 100
        self.current_step = 0
        self.last_write = 0
        self.x_k = len(self.charset)
        self.z_k = 60
        self.observation_space = spaces.Box(-bounds, bounds, (self.x_k + self.z_k + 3,))
        self.d = wgan_sequential_discriminator(self.x_k, self.max_len)
        self.iteration = 0
        self.fake_average = 0
        self.real_average = 0

    def _reset(self):
        self.iteration += 1
        self.sequence = sequence()
        self.current_step = 0
        self.last_write = 0
        self.pointer = 0
        self.output = []
        observation = self._observation()
        return observation

    def _observation(self):
        current_symbol = self.sequence[self.pointer]
        obs = np.concatenate((
            to_categorical(current_symbol, self.z_k).reshape((-1)),
            to_categorical(self.last_write, self.x_k + 1).reshape((-1)),
            [self.current_step, self.pointer]), axis=0)
        # print("Obs shape: {}".format(obs.shape))
        # print("Obs space shape: {}".format(self.observation_space.shape))
        return obs

    def word_vector(self):
        i = np.random.randint(0, self.wordcount)
        word = self.words[i]
        seq = [self.charmap[c] for c in word]
        return self.to_vector(seq)

    def train_d(self, nb_epoch=1, force=False):
        n = len(self.outputs)
        if force or (n >= self.d_warmup):
            xfake = np.vstack(self.outputs)
            xreal = np.vstack(self.samples)
            x = np.vstack((xfake, xreal))
            y = np.vstack((np.ones((n, 1)) * -1, np.ones((n, 1))))
            data = np.hstack((x, y))
            np.random.shuffle(data)
            x = data[:, :-1]
            y = data[:, -1:]
            history = self.d.fit([x], [y], nb_epoch=nb_epoch, verbose=1)
            logging.info("Trained D {} epochs: {}".format(nb_epoch, history.history))
        if n > 1:
            self.fake_average = np.mean(self.d.predict(np.vstack(self.outputs)), axis=None)
            self.real_average = np.mean(self.d.predict(np.vstack(self.samples)), axis=None)

            logging.info("Calculated average fake: {}, average real: {}".format(self.fake_average, self.real_average))

    def to_vector(self, seq):
        assert (len(seq) <= self.max_len)
        seq = [s + 1 for s in seq]
        while (len(seq) < self.max_len):
            seq = [0] + seq
        return np.array(seq)

    def to_string(self, seq):
        return "".join(self.charset[s] for s in seq)

    def _step(self, action):
        self.current_step += 1
        done = False
        reward = 0.0
        if action == 0:
            self.pointer = max(self.pointer - 1, 0)
        elif action == 1:
            self.pointer = min(self.pointer + 1, len(self.sequence) - 1)
        elif action == 2:
            done = True
            reward += 20.0
        else:
            char = action - 3
            assert (char >= 0)
            assert (char < len(self.charset))
            self.output.append(char)
            self.last_write = char + 1
        if len(self.output) >= self.max_len:
            done = True
        if self.current_step >= self.max_steps - 1:
            done = True
        if done:
            if len(self.output) == 0:
                reward -= 1000.0
            else:
                ar = self.to_vector(self.output)
                self.outputs.append(ar)
                output_str = self.to_string(self.output)
                self.samples.append(self.word_vector())
                if len(self.outputs) > self.memory_limit:
                    self.outputs.pop(0)
                if len(self.samples) > self.memory_limit:
                    self.samples.pop(0)
                self.train_d()
                yfake = self.d.predict(ar.reshape((1, -1)))[0, 0]

                #reward += (2*yfake) - self.real_average + self.fake_average
                reward += yfake - self.real_average
                logging.info(
                    "Iteration: {}, Yfake(average): {}, Yreal(average): {}, Yfake: {}, Reward: {}, Output: {}".format(
                        self.iteration, self.fake_average,
                        self.real_average, yfake,
                        float(reward), output_str))
        observation = self._observation()
        info = {}
        return observation, reward, done, info
