import os

os.environ["THEANO_FLAGS"] = "mode=FAST_COMPILE,device=cpu,floatX=float32"
import theano
import theano.tensor as T

import numpy as np

x = np.random.randint(0, 4, (5, 10))
print x

_x = T.imatrix()
c = T.eq(_x, 1)
x2 = c*10+(1-c)*_x
f = theano.function([_x], [c])
f2 = theano.function([_x], [x2])
print(f(x)[0])
print(f2(x)[0])

