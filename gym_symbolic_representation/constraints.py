import numpy as np
from keras.constraints import Constraint
import keras.backend as K


class ClipConstraint(Constraint):
    def __init__(self, min_value, max_value=None):
        if max_value:
            self.min_value = np.float32(min_value)
            self.max_value = np.float32(max_value)
        else:
            self.min_value = np.float32(-min_value)
            self.max_value = np.float32(min_value)

    def __call__(self, p):
        return K.clip(p, self.min_value, self.max_value)


class L1Constraint(Constraint):
    def __init__(self, max_value):
        self.max_value = np.float32(max_value)

    def __call__(self, p):
        l1 = K.sum(K.abs(p), axis=0)
        scale = K.clip(1.0/l1, 0, 1)
        #scale = K.switch(self.max_value > l1, 1.0, 1.0 / l1)
        return p * scale
