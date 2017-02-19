#import os

#os.environ["THEANO_FLAGS"] = "mode=FAST_COMPILE,device=cpu,floatX=float32"

from gym_symbolic_representation.scripts import train
from gym_symbolic_representation.models import dqn
import logging.config

ENV_NAME='Word-WGAN-Combined-v0'

def main():
    logging.config.fileConfig('logging.conf')
    cmd = ["--train", "--steps", "1000000"]
    train.main(cmd, default_path="output/word_wgan_combined/dqn.h5", env_name=ENV_NAME, create_agent=dqn.create_agent_dqn)


if __name__ == "__main__":
    main()
