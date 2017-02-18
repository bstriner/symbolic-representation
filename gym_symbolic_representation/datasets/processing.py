import pickle
import os
from ..utils import clean_text


def word_counts(ws):
    dict = {}
    for word in ws:
        if word not in dict:
            dict[word] = 1
        else:
            dict[word] += 1
    return dict


def character_counts(ws):
    dict = {}
    for word in ws:
        for char in word:
            if char not in dict:
                dict[char] = 1
            else:
                dict[char] += 1
    return dict


def process(ws):
    ws = [clean_text(w) for w in ws]
    wc = word_counts(ws)
    cc = character_counts(ws)

    print("Word count: {}".format(len(ws)))
    print("Unique words: {}".format(len(wc)))
    print("Unique characters: {}".format(len(cc)))
    print("Longest word: {}".format(max(len(w) for w in ws)))
    print("Shorted word: {}".format(min(len(w) for w in ws)))

    charset = list(cc.keys())
    charset.sort()
    charmap = {k: i for i, k in enumerate(charset)}

    print("Charset: {}".format(charset))

    data = {"words": ws, "characters": cc, "charset": charset, "charmap": charmap}
    return data


def save(data, path):
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    with open(path, 'wb') as f:
        pickle.dump(data, f)


def load(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def load_or_create(path, words):
    if os.path.exists(path):
        return load(path)
    else:
        data = process(words())
        save(data, path)
        return data
