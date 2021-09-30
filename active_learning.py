import argparse
import yaml

from utils.utils import load_data
import IPython


def main():
    """Main funciton to run"""
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--config_path", default="config.yaml")
    args = parser.parse_args()
    config = yaml.safe_load(open(args.config_path, "r"))

    # the data, shuffled and split between train and test sets
    (X_train_All, y_train_All), (X_test, y_test) = load_data(config["data"]["dataset"])

    print("Original size of the dataset")
    print("X_train shape:", X_train_All.shape)
    print("y_train shape:", y_train_All.shape)
    print("X_test shape:", X_test.shape)
    print("y_test shape:", y_test.shape)


if __name__ == "__main__":
    main()
