from nltk.corpus import shakespeare
import itertools
def words():
    return list(itertools.chain.from_iterable(shakespeare.words(fileid) for fileid in shakespeare.fileids()))