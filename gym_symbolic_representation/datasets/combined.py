from nltk.corpus import brown, reuters, gutenberg


def words():
    return list(brown.words()) + list(reuters.words()) + list(gutenberg.words())
