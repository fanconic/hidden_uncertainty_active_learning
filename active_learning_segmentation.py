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
from src.utils.metrics import IoU
import IPython
from copy import deepcopy
from pprint import pprint
from src.data.sampling import MapDataset
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
import os
from src.utils.array_utils import to_label_tensor, mask_to_class, mask_to_rgb
from torchvision.utils import save_image
import itertools


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
    train_target_transform_list = []
    test_transform_list = []
    test_target_transform_list = []

    resize = (config["data"]["img_rows"], config["data"]["img_cols"])

    mapping = {
        0: 0,  # unlabeled
        1: 0,  # ego vehicle
        2: 0,  # rect border
        3: 0,  # out of roi
        4: 0,  # static
        5: 0,  # dynamic
        6: 0,  # ground
        7: 1,  # road
        8: 0,  # sidewalk
        9: 0,  # parking
        10: 0,  # rail track
        11: 0,  # building
        12: 0,  # wall
        13: 0,  # fence
        14: 0,  # guard rail
        15: 0,  # bridge
        16: 0,  # tunnel
        17: 0,  # pole
        18: 0,  # polegroup
        19: 0,  # traffic light
        20: 0,  # traffic sign
        21: 0,  # vegetation
        22: 0,  # terrain
        23: 2,  # sky
        24: 0,  # person
        25: 0,  # rider
        26: 3,  # car
        27: 3,  # truck
        28: 3,  # bus
        29: 3,  # caravan
        30: 3,  # trailer
        31: 3,  # train
        32: 3,  # motorcycle
        33: 3,  # bicycle
        -1: 0,  # licenseplate
    }

    if config["data"]["augmentation"]:
        train_transform_list.extend(
            [
                transforms.Resize(resize, interpolation=2),
                transforms.RandomCrop(resize, padding=10),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )
        train_target_transform_list.extend(
            [
                transforms.Resize(resize, interpolation=0),
                transforms.RandomCrop(resize, padding=10),
                transforms.RandomHorizontalFlip(),
                transforms.Lambda(to_label_tensor),
                transforms.Lambda(lambda x: mask_to_class(x, mapping)),
            ]
        )
    else:
        train_transform_list.extend(
            [transforms.Resize(resize, interpolation=2), transforms.ToTensor()]
        )
        train_target_transform_list.extend(
            [
                transforms.Resize(resize, interpolation=0),
                transforms.Lambda(to_label_tensor),
                transforms.Lambda(lambda x: mask_to_class(x, mapping)),
            ]
        )

    test_transform_list.extend(
        [transforms.Resize(resize, interpolation=2), transforms.ToTensor()]
    )
    test_target_transform_list.extend(
        [
            transforms.Resize(resize, interpolation=0),
            transforms.Lambda(to_label_tensor),
            transforms.Lambda(lambda x: mask_to_class(x, mapping)),
        ]
    )

    if config["data"]["rgb_normalization"]:
        normalize = transforms.Normalize(
            mean=config["data"]["mean"],
            std=config["data"]["std"],
        )

        train_transform_list.append(normalize)
        test_transform_list.append(normalize)

    train_transform = transforms.Compose(train_transform_list)
    train_target_transform = transforms.Compose(train_target_transform_list)
    test_transform = transforms.Compose(test_transform_list)
    test_target_transform = transforms.Compose(test_target_transform_list)

    # in the City Scapes dataset, the test data set corresponds to the original validation dataset.
    # The new validation dataset is computed by taking a split
    train_whole, test_ds = load_data(
        config["data"]["dataset"],
        train_transform=None,
        test_transform=test_transform,
        path=config["data"]["path"],
        train_target_transform=None,
        test_target_transform=test_target_transform,
    )

    # obtain training indices that will be used for validation
    num_train = len(train_whole)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(np.floor(config["data"]["val_size"] * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]

    train_subs = torch.utils.data.Subset(train_whole, train_idx)
    val_subs = torch.utils.data.Subset(train_whole, valid_idx)
    train_ds = MapDataset(
        train_subs, train_transform, target_map_fn=train_target_transform
    )
    val_ds = MapDataset(val_subs, test_transform, target_map_fn=test_target_transform)

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
    criterion = nn.CrossEntropyLoss(ignore_index=config["data"]["ignore_label"])

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
        reduction=config["training"]["reduction"],
    )

    wrapper = ModelWrapper(models=models, criterion=criterion, heuristic=heuristic)
    wrapper.add_metric(
        "iou",
        lambda: IoU(
            num_classes=config["data"]["nb_classes"],
            ignore_label=config["data"]["ignore_label"],
        ),
    )

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

    test_ious = []
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
                "train_iou": wrapper.metrics["train_iou"].value,
                "val_iou": wrapper.metrics["val_iou"].value,
                "test_iou": wrapper.metrics["test_iou"].value,
            }
        )

        # Log progress
        test_ious.append(wrapper.metrics["test_iou"].value)
        test_losses.append(test_loss)
        samples.append(len(al_dataset))

        flag = al_loop.step()
        if not flag:
            # We are done labelling! stopping
            break

    if config["save_plot"]:
        plt.plot(samples, test_ious)
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
        df["test_ious_run{}".format(str(run + 1))] = test_ious
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
