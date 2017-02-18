from keras.initializations import uniform


def uniform_init(scale=1e-2):
    def init(shape, name=None, dim_ordering='th'):
        return uniform(shape, scale, name, dim_ordering)

    return init
