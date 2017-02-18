from gym import Env
from gym.utils import seeding
from gym import spaces
import numpy as np
from ..datasets import combined
from ..datasets.processing import load_or_create
from ..priors.sequence import sequence
from ..discriminators.wgan_sequential import wgan_sequential_discriminator


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
        self.max_steps = 100
        self.max_len = 50
        self.pointer = 0
        self.sequence = sequence()
        self.output = []
        self.action_space = spaces.Discrete(3 + len(self.charset))
        bounds = float('inf')
        self.observation_space = spaces.Box(-bounds, bounds, (4,))
        self.viewer = None
        self.outputs = []
        self.samples = []
        self.memory_limit = 5000
        self.d = wgan_sequential_discriminator(len(self.charset), self.max_len)
        self.d_warmup = 100

    def _observation(self):
        pass

    def word_vector(self):
        i = np.random.randint(0, self.wordcount)
        word = self.words[i]
        seq = [self.charmap[c] for c in word]
        return self.to_vector(seq)

    def _train_d(self):
        n = len(self.output)
        if n >= self.d_warmup:
            xfake = np.vstack(self.outputs)
            xreal = np.vstack(self.samples)
            x = np.vstack((xfake, xreal))
            y = np.vstack((np.ones((n, 1)) * -1, np.ones((n, 1))))
            data = np.hstack((x, y))
            np.random.shuffle(data)
            x = data[:,:-1]
            y = data[:,-1:]
            self.d.fit([x], [y], nb_epoch=5, verbose=1)

    def to_vector(self, seq):
        seq = [s + 1 for s in seq]
        while (len(seq) < self.max_len):
            seq = [0] + seq
        return np.array(seq)

    def _step(self, action):
        done = False
        reward = 0.0
        if action == 0:
            self.pointer = max(self.pointer - 1, 0)
        elif action == 1:
            self.pointer = min(self.pointer + 1, len(self.sequence) - 1)
        elif action == 2:
            done = True
        else:
            char = action - 3
            assert (char >= 0)
            assert (char < len(self.charset))
            self.output.append(char)
        if len(self.output) > self.max_len:
            done = True
        if done:
            ar = self.to_vector(self.output)
            self.outputs.append(ar)
            self.samples.append(self.word_vector())
            self.output = []
            if len(self.outputs) > self.memory_limit:
                self.outputs.pop(0)
            if len(self.samples) > self.memory_limit:
                self.samples.pop(0)
            self._train_d()
            reward = self.d.predict(ar.reshape((1, -1)))[0]
        observation = self._observation()
        info = {}
        return observation, reward, done, info
