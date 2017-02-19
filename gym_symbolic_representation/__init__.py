from gym.envs.registration import register
from .envs.word_wgan_env import WordWganEnv

register(
    id='Word-WGAN-Combined-v0',
    entry_point='gym_symbolic_representation.envs:WordWganCombinedEnv',
    nondeterministic=True
)
register(
    id='Word-WGAN-Shakespeare-v0',
    entry_point='gym_symbolic_representation.envs:WordWganShakespeareEnv',
    nondeterministic=True
)
