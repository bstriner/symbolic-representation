from gym.envs.registration import register
from .envs.word_wgan_env import WordWganEnv

register(
    id='Word-WGAN-v0',
    entry_point='gym_symbolic_representation.envs.word_wgan_env:WordWganEnv',
    nondeterministic=True
)
