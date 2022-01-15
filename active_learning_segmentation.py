# Parts taken from https://github.com/ElementAI/baal/blob/master/experiments/mlp_mcdropout.py

import argparse
import yaml
import torch
from src.utils.utils import (
    load_data,
    get_model,
    get_heuristic,
    get_optimizer,
    get_scheduler,
)
from src.data.dataset import ActiveLearningDataset
from src.layers.consistent_dropout import patch_module
from src.models.model_wrapper import ModelWrapper
from src.active.active_loop import ActiveLearningLoop
from torch import nn, optim
from src.utils.metrics import IoU, AUROC
import IPython
from copy import deepcopy
from pprint import pprint
from src.data.sampling import MapDataset
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
import os
from torchvision.utils import save_image
import src.data.preprocessing as transforms
import wandb


def set_seed(seed):
    """Set all random seeds"""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    g = torch.Generator()
    g.manual_seed(seed)
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

    if config["data"]["resize"]:
        resize_factor = (config["data"]["img_cols"], config["data"]["img_rows"])
        train_transform_list.append(transforms.Resize(resize_factor))
        test_transform_list.append(transforms.Resize(resize_factor))

    if config["data"]["augmentation"]:
        train_transform_list.extend(
            [
                # transforms.RandomRotate(10),
                # transforms.RandomScale(2),
                transforms.RandomCrop(config["training"]["crop_size"]),
                transforms.RandomHorizontalFlip(),
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

    # in the City Scapes dataset, the test data set corresponds to the original validation dataset.
    # The new validation dataset is computed by taking a split
    train_whole, test_ds = load_data(
        config["data"]["dataset"],
        train_transform=None,
        test_transform=test_transform,
        path=config["data"]["path"],
    )

    # obtain training indices that will be used for validation
    num_train = len(train_whole)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(np.floor(config["data"]["val_size"] * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]

    train_subs = torch.utils.data.Subset(train_whole, train_idx)
    val_subs = torch.utils.data.Subset(train_whole, valid_idx)
    train_ds = MapDataset(train_subs, train_transform, segmentation=True)
    val_ds = MapDataset(val_subs, test_transform, segmentation=True)

    al_dataset = ActiveLearningDataset(
        train_ds,
        # pool_specifics={"transform": test_transform},
        random_state=random_state,
        pool_specifics={"map": test_transform},
    )
    al_dataset.label_randomly(
        config["training"]["initially_labelled"],
        balanced=config["training"]["initially_balanced"],
        classes=list(range(config["data"]["nb_classes"])),
    )

    # Loss
    criterion = nn.CrossEntropyLoss(ignore_index=config["data"]["ignore_label"])

    # Create MLPs to classify MNIST
    models = []

    # Get (emsemble) models
    for _ in range(config["model"]["ensemble"]):
        model = get_model(config["model"])
        if config["model"]["mc_dropout"]:
            model = patch_module(model)  # Set dropout layers for MC-Dropout.
        if use_cuda:
            model = model.cuda()
        models.append(model)
    models = tuple(models)

    # Define heuristics
    heuristic = get_heuristic(
        config["training"]["heuristic"],
        random_state=random_state,
        shuffle_prop=config["training"]["shuffle_prop"],
        reduction=config["training"]["reduction"],
    )

    wrapper = ModelWrapper(
        models=models,
        criterion=criterion,
        heuristic=heuristic,
    )
    wrapper.add_metric(
        "iou",
        lambda: IoU(
            num_classes=config["data"]["nb_classes"],
        ),
    )
    wrapper.add_metric(
        "auroc",
        lambda: AUROC(
            num_classes=config["data"]["nb_classes"],
            heuristic=heuristic,
            ignore_label=config["data"]["ignore_label"],
        ),
        train=False,
        val=False,
    )

    # Setup our active learning loop for our experiments
    al_loop = ActiveLearningLoop(
        dataset=al_dataset,
        get_probabilities=wrapper.predict_on_dataset_generator,
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

    samples = []
    test_ious = []
    test_losses = []
    train_ious = []
    train_losses = []
    val_ious = []
    val_losses = []

    wandb.watch(models, criterion=criterion)

    # Set initial model weights
    # reset the learning rate scheduler
    # reset the optimizer
    for step in range(config["training"]["iterations"]):
        optimizers = []
        schedulers = []
        for i, model in enumerate(models):
            # load initial model weights
            model.load_state_dict(initial_weights[i])

            # set optimizer
            optimizer = get_optimizer(model, config)

            # set scheduler:
            scheduler = get_scheduler(optimizer, config)

            optimizers.append(optimizer)
            schedulers.append(scheduler)

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
            workers=3,
            al_iteration=step,
            average_predictions=1,
        )

        if config["training"]["load_best_model"]:
            wrapper.load_state_dict(best_weights)

        test_loss = wrapper.test_on_dataset(
            test_ds,
            batch_size=config["training"]["batch_size"],
            use_cuda=use_cuda,
            average_predictions=config["model"]["mc_iterations"],
            workers=3,
        )

        logs = {
            "iteration": step,
            "dataset_size": len(al_dataset),
            "end_test_loss": wrapper.metrics["test_loss"].value,
            "end_test_iou": wrapper.metrics["test_iou"].value,
            "end_test_auroc": wrapper.metrics["test_auroc"].value,
        }

        pprint(logs)
        wandb.log(logs)

        # Log progress
        samples.append(len(al_dataset))
        test_ious.append(wrapper.metrics["test_iou"].value)
        test_losses.append(test_loss)
        train_ious.append(wrapper.metrics["train_iou"].value)
        train_losses.append(wrapper.metrics["train_loss"].value)
        val_ious.append(wrapper.metrics["val_iou"].value)
        val_losses.append(wrapper.metrics["val_loss"].value)

        flag = al_loop.step()
        if not flag:
            # We are done labelling! stopping
            break

    if config["save_plot"]:
        plt.plot(samples, test_ious)
        plt.grid()
        plt.title("Experiment {}".format(config["name"]))
        plt.savefig(
            "experiment_outputs/final_runs/{}/run_{}.pdf".format(
                config["name"], str(run + 1)
            ),
            format="pdf",
            bbox_inches="tight",
        )

    if config["save_df"]:
        df = pd.DataFrame()
        df["samples"] = samples
        df = df.set_index("samples")
        df["test_ious_run{}".format(str(run + 1))] = test_ious
        df["test_losses_run{}".format(str(run + 1))] = test_losses
        df["train_ious_run{}".format(str(run + 1))] = train_ious
        df["train_losses_run{}".format(str(run + 1))] = train_losses
        df["val_ious_run{}".format(str(run + 1))] = val_ious
        df["val_losses_run{}".format(str(run + 1))] = val_losses
        return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--config_path", default="config.yaml")
    args = parser.parse_args()
    config = yaml.safe_load(open(args.config_path, "r"))
    print(config["name"])
    df = pd.DataFrame()

    mode = "online" if config["wandb_logging"] else "disabled"

    for run in range(config["runs"]):
        wandb.init(
            mode=mode,
            project="hidden_uncertainty",
            entity="fanconic",
            name=config["name"] + "_run{}".format(run + 1),
            reinit=True,
            config=config,
        )
        run_df = main(config, run, config["random_state"][run])
        df = df.join(run_df, how="right")

    df.to_csv("experiment_outputs/final_runs/{}.csv".format(config["name"]))
