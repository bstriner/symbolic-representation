import os

os.environ["THEANO_FLAGS"] = "mode=FAST_COMPILE,device=cpu,floatX=float32"

from gym_symbolic_representation.scripts import train
from gym_symbolic_representation.models import dqn


def main():
    cmd = ["--train", "--steps", "100000"]
    train.main(cmd, default_path="output/word_wgan/dqn.h5", env_name="Word-WGAN-v0", create_agent=dqn.create_agent_dqn)


if __name__ == "__main__":
    main()
