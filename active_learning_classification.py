# Parts taken from https://github.com/ElementAI/baal/blob/master/experiments/mlp_mcdropout.py

import argparse
import yaml
import torch
from torchvision import transforms
from src.utils.utils import load_data, get_model, get_heuristic
from src.data.dataset import ActiveLearningDataset
from src.layers.consistent_dropout import patch_module
from src.models.model_wrapper import ModelWrapper
from src.active.active_loop import ActiveLearningLoop
from torch import nn, optim
from src.utils.metrics import Accuracy
import IPython
from copy import deepcopy
from pprint import pprint
from src.data.sampling import MapDataset
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
import os


def set_seed(seed):
    """Set all random seeds"""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def main(config, run, random_state):
    """Main funciton to run"""
    use_cuda = torch.cuda.is_available()
    print("Cuda is available: ", use_cuda)

    set_seed(random_state)

    # Load dataset
    train_transform_list = []
    test_transform_list = []
    if config["data"]["augmentation"]:
        train_transform_list.extend(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
            ]
        )
    else:
        train_transform_list.append(transforms.ToTensor())

    test_transform_list.append(transforms.ToTensor())

    if config["data"]["rgb_normalization"]:
        normalize = transforms.Normalize(
            mean=config["data"]["mean"],
            std=config["data"]["std"],
        )

        train_transform_list.append(normalize)
        test_transform_list.append(normalize)

    train_transform = transforms.Compose(train_transform_list)
    test_transform = transforms.Compose(test_transform_list)

    train_whole, test_ds = load_data(
        config["data"]["dataset"], None, test_transform, path=config["data"]["path"]
    )

    # obtain training indices that will be used for validation
    num_train = len(train_whole)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(np.floor(config["data"]["val_size"] * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]

    train_subs = torch.utils.data.Subset(train_whole, train_idx)
    val_subs = torch.utils.data.Subset(train_whole, valid_idx)
    train_ds = MapDataset(train_subs, train_transform)
    val_ds = MapDataset(val_subs, test_transform)

    al_dataset = ActiveLearningDataset(
        train_ds,
        # pool_specifics={"transform": test_transform},
        random_state=random_state,
    )
    al_dataset.label_randomly(
        config["training"]["initially_labelled"],
        balanced=config["training"]["initially_balanced"],
        classes=list(range(config["data"]["nb_classes"])),
    )

    # Loss
    criterion = nn.CrossEntropyLoss()

    # Create MLPs to classify MNIST
    models = []
    optimizers = []
    schedulers = []

    for _ in range(config["model"]["ensemble"]):
        model = get_model(config["model"])
        if config["model"]["mc_dropout"]:
            model = patch_module(model)  # Set dropout layers for MC-Dropout.
        if use_cuda:
            model = model.cuda()

        optimizer = optim.Adam(
            model.parameters(),
            lr=config["optimizer"]["lr"],
            betas=config["optimizer"]["betas"],
            weight_decay=config["optimizer"]["weight_decay"],
        )

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=config["training"]["lr_reduce_factor"],
            patience=config["training"]["patience_lr_reduce"],
        )

        models.append(model)
        optimizers.append(optimizer)
        schedulers.append(scheduler)

    heuristic = get_heuristic(
        config["training"]["heuristic"],
        random_state=random_state,
        shuffle_prop=config["training"]["shuffle_prop"],
    )

    wrapper = ModelWrapper(models=models, criterion=criterion, heuristic=heuristic)
    wrapper.add_metric("accuracy", lambda: Accuracy())

    # Setup our active learning loop for our experiments
    al_loop = ActiveLearningLoop(
        dataset=al_dataset,
        get_probabilities=wrapper.predict_on_dataset,
        heuristic=heuristic,
        ndata_to_label=config["training"]["ndata_to_label"],
        # KWARGS for predict_on_dataset
        iterations=config["model"]["mc_iterations"],
        batch_size=config["training"]["batch_size"],
        use_cuda=use_cuda,
        verbose=config["training"]["verbose"],
    )

    # Following Gal 2016, we reset the weights at the beginning of each step.
    initial_weights = [deepcopy(model.state_dict()) for model in models]
    initial_states = [deepcopy(optimizer.state_dict()) for optimizer in optimizers]

    test_accuracies = []
    test_losses = []
    samples = []

    # Set initial model weights
    # reset the learning rate scheduler
    # reset the optimizer
    for step in range(config["training"]["iterations"]):
        for i, (model, optimizer, scheduler) in enumerate(
            zip(models, optimizers, schedulers)
        ):
            model.load_state_dict(initial_weights[i])

            if isinstance(optimizer, optim.Adam):
                optimizer.load_state_dict(initial_states[i])

            scheduler._reset()

        train_loss, best_weights = wrapper.train_on_dataset(
            al_dataset,
            val_ds,
            optimizers=optimizers,
            schedulers=schedulers,
            batch_size=config["training"]["batch_size"],
            epoch=config["training"]["epochs"],
            use_cuda=use_cuda,
            early_stopping=config["training"]["early_stopping"],
            patience=config["training"]["patience_early_stopping"],
            verbose=config["training"]["verbose"],
            return_best_weights=config["training"]["load_best_model"],
        )

        if config["training"]["load_best_model"]:
            wrapper.load_state_dict(best_weights)

        test_loss = wrapper.test_on_dataset(
            test_ds,
            batch_size=config["training"]["batch_size"],
            use_cuda=use_cuda,
            average_predictions=config["model"]["mc_iterations"],
        )

        pprint(
            {
                "dataset_size": len(al_dataset),
                "train_loss": wrapper.metrics["train_loss"].value,
                "val_loss": wrapper.metrics["val_loss"].value,
                "test_loss": wrapper.metrics["test_loss"].value,
                "train_accuracy": wrapper.metrics["train_accuracy"].value,
                "val_accuracy": wrapper.metrics["val_accuracy"].value,
                "test_accuracy": wrapper.metrics["test_accuracy"].value,
            }
        )

        # Log progress
        test_accuracies.append(wrapper.metrics["test_accuracy"].value)
        test_losses.append(test_loss)
        samples.append(len(al_dataset))

        flag = al_loop.step()
        if not flag:
            # We are done labelling! stopping
            break

    if config["save_plot"]:
        plt.plot(samples, test_accuracies)
        plt.grid()
        plt.title("Experiment {}".format(config["name"]))
        plt.savefig(
            "experiment_outputs/week{}/{}/run_{}.pdf".format(
                str(config["week"]), config["name"], str(run + 1)
            ),
            format="pdf",
            bbox_inches="tight",
        )

    if config["save_df"]:
        df = pd.DataFrame()
        df["samples"] = samples
        df = df.set_index("samples")
        df["test_accuracy_run{}".format(str(run + 1))] = test_accuracies
        df["test_losses_run{}".format(str(run + 1))] = test_losses
        return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--config_path", default="config.yaml")
    args = parser.parse_args()
    config = yaml.safe_load(open(args.config_path, "r"))
    print(config["name"])
    df = pd.DataFrame()

    for run in range(config["runs"]):
        run_df = main(config, run, config["random_state"][run])
        df = df.join(run_df, how="right")

    df.to_csv(
        "experiment_outputs/week{}/{}.csv".format(str(config["week"]), config["name"])
    )
