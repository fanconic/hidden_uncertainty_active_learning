# Parts taken from https://github.com/ElementAI/baal/blob/master/experiments/mlp_mcdropout.py

import argparse
import yaml
import torch
from torchvision import transforms
from src.utils.utils import load_data, get_model
from src.data.dataset import ActiveLearningDataset
from src.layers.consistent_dropout import patch_module
from src.models.model_wrapper import ModelWrapper
from src.active.active_loop import ActiveLearningLoop
from torch import nn, optim
from src.active.heuristics import BALD
from src.utils.metrics import Accuracy
import IPython
from copy import deepcopy
from pprint import pprint
from src.data.sampling import sampleFromClass
import matplotlib.pyplot as plt


def main():
    """Main funciton to run"""

    use_cuda = torch.cuda.is_available()
    print("Cude is available: ", use_cuda)

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--config_path", default="config.yaml")
    args = parser.parse_args()
    config = yaml.safe_load(open(args.config_path, "r"))

    # Load dataset
    train_transform = transforms.ToTensor()
    test_transform = transforms.ToTensor()
    train_ds, test_ds = load_data(
        config["data"]["dataset"], train_transform, test_transform
    )

    # validation split
    val_ds, train_ds = sampleFromClass(
        train_ds, config["data"]["val_size"] // config["data"]["nb_classes"]
    )

    al_dataset = ActiveLearningDataset(
        train_ds,
        # pool_specifics={"transform": test_transform},
        random_state=config["random_state"],
    )
    al_dataset.label_randomly(
        config["training"]["initially_labelled"],
        balanced=True,
        classes=list(range(config["data"]["nb_classes"])),
    )  # Start with 200 items labelled.

    # Loss
    criterion = nn.CrossEntropyLoss()

    # Create MLPs to classify MNIST
    models = []
    optimizers = []
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

        models.append(model)
        optimizers.append(optimizer)

    wrapper = ModelWrapper(models=models, criterion=criterion)
    wrapper.add_metric("accuracy", lambda: Accuracy())

    # We will use BALD as our heuristic as it is a great tradeoff between performance and efficiency.
    bald = BALD()

    # Setup our active learning loop for our experiments
    al_loop = ActiveLearningLoop(
        dataset=al_dataset,
        get_probabilities=wrapper.predict_on_dataset,
        heuristic=bald,
        ndata_to_label=config["training"][
            "ndata_to_label"
        ],  # We will label 100 examples per step.
        # KWARGS for predict_on_dataset
        iterations=config["model"]["mc_iterations"],
        batch_size=config["training"]["batch_size"],
        use_cuda=use_cuda,
        verbose=config["training"]["verbose"],
    )

    # Following Gal 2016, we reset the weights at the beginning of each step.
    initial_weights = [deepcopy(model.state_dict()) for model in models]

    test_accuracies = []
    test_losses = []
    samples = []

    for step in range(config["training"]["iterations"]):
        for i, model in enumerate(models):
            model.load_state_dict(initial_weights[0])

        train_loss = wrapper.train_on_dataset(
            al_dataset,
            val_ds,
            optimizers=optimizers,
            batch_size=config["training"]["batch_size"],
            epoch=config["training"]["epochs"],
            use_cuda=use_cuda,
            patience=config["training"]["patience"],
            verbose=config["training"]["verbose"],
        )

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
                "test_loss": wrapper.metrics["test_loss"].value,
                "train_accuracy": wrapper.metrics["train_accuracy"].value,
                "test_accuracy": wrapper.metrics["test_accuracy"].value,
                "val_accuracy": wrapper.metrics["val_accuracy"].value,
                "val_accuracy": wrapper.metrics["val_accuracy"].value,
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
            "experiment_outputs/{}.pdf".format(config["name"]),
            format="pdf",
            bbox_inches="tight",
        )


if __name__ == "__main__":
    main()
