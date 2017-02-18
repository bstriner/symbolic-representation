import numpy as np


def exponential_sampling(n, k, rate=0.98):
    x = np.arange(k)
    t = np.power(rate, x)
    p = t / np.sum(t)
    c = np.cumsum(p)
    r = np.random.uniform(0.0, 1.0, (n,))
    c = np.repeat(c.reshape((1, -1)), n, axis=0)
    r = np.repeat(r.reshape((-1, 1)), k, axis=1)
    ret = np.int32(np.sum(c < r, axis=1))
    return ret


def sample_length(min_limit, max_limit, rate=0.98):
    x = exponential_sampling(1, max_limit-min_limit, rate=rate)
    return min_limit + x


def sequence(min_limit=5, max_limit=15, k=60, len_rate=0.8, word_rate=0.92):
    sample_len = sample_length(min_limit, max_limit, rate=len_rate)
    words = exponential_sampling(sample_len, k, rate=word_rate)
    return words


if __name__ == "__main__":

    print("Samples")
    x = exponential_sampling(5000, 60, rate=0.92)
    a = np.min(x)
    b = np.max(x)
    for i in range(a, b + 1):
        print("{}: {}".format(i, len(list(z for z in x if z == i))))


    print("Lengths")
    x = [sample_length(5, 15, rate=0.8) for _ in range(1000)]
    a = np.min(x)
    b = np.max(x)
    for i in range(a, b + 1):
        print("{}: {}".format(i, len(list(z for z in x if z == i))))

    print("Sequences")
    for i in range(20):
        print sequence()
