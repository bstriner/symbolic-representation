from .word_wgan_env import WordWganEnv
from ..datasets import shakespeare
from ..datasets.processing import load_or_create
class WordWganCombinedEnv(WordWganEnv):
    """

    Actions:
        0: move left
        1: move right
        2: done
        3+: output characters

    """

    def __init__(self):
        path = "output/words-combined.pkl"
        data = load_or_create(path, shakespeare.words)
        WordWganEnv.__init__(self, data=data, max_len=50, max_steps=150)

