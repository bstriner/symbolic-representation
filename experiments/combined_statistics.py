from gym_symbolic_representation.datasets import combined
from gym_symbolic_representation.datasets.processing import process, save


def main():
    ws = combined.words()
    data = process(ws)
    save(data, "output/words.pkl")


if __name__ == "__main__":
    main()
