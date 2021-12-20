# Taken from https://github.com/ElementAI/baal/blob/a9cc0034c40d0541234a3c27ff5ccbd97278bcb3/baal/modelwrapper.py#L30

from numpy.lib.arraysetops import isin
from torch import nn, optim
import sys
from collections.abc import Sequence
from copy import deepcopy
from typing import Callable, Optional
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import structlog
import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import default_collate
from tqdm import tqdm
import torchvision
import torch.nn.functional as F

from src.utils.array_utils import stack_in_memory
from src.utils.cuda_utils import to_cuda
from src.utils.iterutils import map_on_tensor
from src.utils.metrics import AUROC, PAC, Loss

from src.models.BNN import BNN
from src.models.BCNN import BCNN
from src.models.MIR import MIR
from src.models.UNet import UNet
from src.active.heuristics import Precomputed
from src.utils.utils import CITYSCAPE_PALETTE, fig2img, addlabels
import random
import wandb

log = structlog.get_logger("ModelWrapper")


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def extract_features(
    model, dataset, cuda=False, return_true_labels=False, segmentation=False
):
    """extracts features and predictions of the MIR model
    Args:
        model (nn.Model): MIR model
        dataset: training dataset
        cuda (bool): use cuda
        return_true_labels (bool): if true, returns the true labels, not predicted ones
        segmentation (bool): if we are trying to solve a segmentation problem
    returns:
        features, predictions
    """
    features = []
    predictions = []
    for i, batch in enumerate(dataset):
        x, y = batch

        if cuda:
            x = to_cuda(x)

        out = model(x, return_features=True)
        feature = out["features"]

        if not segmentation:
            output = F.softmax(out["prediction"], 1)
            # Flatten, if they are multidimensional
            if len(feature.shape) > 2:
                feature = torch.flatten(feature, 1)

            features.append(feature.cpu().detach())

            if return_true_labels:
                predictions.append(y.cpu().detach())
            else:
                predictions.append(np.argmax(output.cpu().detach(), axis=1))

        else:
            output = F.softmax(out["features"], 1)
            # rearrange the dimensions
            feature = feature.permute((0, 2, 3, 1)).flatten(0, 2)
            prediction = torch.argmax(output.permute(0, 2, 3, 1), -1).flatten(0, 2)

            features.append(feature.cpu().detach())
            predictions.append(prediction.cpu().detach())

    features = np.concatenate(features, axis=0)
    predictions = np.concatenate(predictions, axis=0)
    return features, predictions


def _stack_preds(out):
    if isinstance(out[0], Sequence):
        out = [torch.stack(ts, dim=-1) for ts in zip(*out)]
    else:
        out = torch.stack(out, dim=-1)
    return out


class ModelWrapper:
    """
    Wrapper created to ease the training/testing/loading.
    Args:
        model (nn.Module): The model to optimize.
        criterion (Callable): A loss function.
        replicate_in_memory (bool): Replicate in memory optional.
    """

    def __init__(
        self,
        models,
        criterion,
        replicate_in_memory=True,
        heuristic=None,
    ):
        self.models = models
        self.criterion = criterion
        self.heuristic = heuristic
        self.metrics = dict()
        self.track_metric = []
        self.add_metric("loss", lambda: Loss(), track_metric=False)
        self.replicate_in_memory = replicate_in_memory

    def add_metric(
        self,
        name: str,
        initializer: Callable,
        train=True,
        test=True,
        val=True,
        track_metric=True,
    ):
        """
        Add a baal.utils.metric.Metric to the Model.
        Args:
            name (str): name of the metric.
            initializer (Callable): lambda to initialize a new instance of a
                                    baal.utils.metrics.Metric object.
            train (bool): log training
            test (bool): log test
            val (bool): log validation
            track_metric (bool): if the metric is tracked
        """
        if track_metric:
            self.track_metric.append(name)  # used if verbose during training
        if test:
            self.metrics["test_" + name] = initializer()
        if train:
            self.metrics["train_" + name] = initializer()
        if val:
            self.metrics["val_" + name] = initializer()

    def _reset_metrics(self, filter=""):
        """
        Reset all Metrics according to a filter.
        Args:
            filter (str): Only keep the metric if `filter` in the name.
        """
        for k, v in self.metrics.items():
            if filter in k:
                v.reset()

    def _update_metrics(
        self, out, target, loss, filter="", reduce=False, uncertainty=None
    ):
        """
        Update all metrics.
        Args:
            out (Tensor): Prediction.
            target (Tensor): Ground truth.
            loss (Tensor): Loss from the criterion.
            filter (str): Only update metrics according to this filter.
            reduce (bool): if the full iterations (MC, DE) are passed, without taking the mean
            uncertainty (Tensor): precomputed uncertainties
        """
        if reduce:
            out_unreduced = deepcopy(out)
            out = out.mean(-1)
        for k, v in self.metrics.items():
            if filter in k:
                if "loss" in k:
                    v.update(loss)
                else:
                    if (
                        isinstance(v, (PAC, AUROC))
                        and reduce
                        and not isinstance(self.heuristic, Precomputed)
                    ):
                        v.update(out_unreduced, target)
                    elif (
                        isinstance(v, (PAC, AUROC))
                        and reduce
                        and isinstance(self.heuristic, Precomputed)
                    ):
                        v.update(out_unreduced, target, uncertainty=uncertainty)
                    elif isinstance(v, (AUROC, PAC)) and not reduce:
                        raise ValueError
                    else:
                        v.update(out, target)

    def train_on_dataset(
        self,
        dataset,
        val_dataset,
        optimizers,
        schedulers,
        batch_size,
        epoch,
        use_cuda,
        workers=4,
        early_stopping=False,
        collate_fn: Optional[Callable] = None,
        regularizer: Optional[Callable] = None,
        verbose: bool = True,
        patience: int = None,
        average_predictions: int = 1,
        return_best_weights: bool = False,
        al_iteration: int = 1,
    ):
        """
        Train for `epoch` epochs on a Dataset `dataset.
        Args:
            dataset (Dataset): Pytorch Dataset to be trained on.
            val_dataset (Dataset): Pytorch Dataset, for the model to be evaluated on
            optimizer (optim.Optimizer): list of Optimizer to use.
            schedulers (optim.lr_scheduler): list of Schedulers
            batch_size (int): The batch size used in the DataLoader.
            epoch (int): Number of epoch to train for.
            use_cuda (bool): Use cuda or not.
            workers (int): Number of workers for the multiprocessing.
            early_stopping (bool): early stopping option if validation doesn't improve
            collate_fn (Optional[Callable]): The collate function to use.
            regularizer (Optional[Callable]): The loss regularization for training.
            verbose (bool): show training progress
            patience (int): patience epochs, as long as the model did not improve
            average_predictions (int): average predictions for validation dataset
            return_best_weights (bool): if the best weights shall be returned
            al_iteration: current iteration of the active learning process
        Returns:
            The training history.
        """
        history = []
        log.info("Starting training", epoch=epoch, dataset=len(dataset))
        collate_fn = collate_fn or default_collate
        best_loss = np.inf
        patience_counter = 0
        best_weights = None
        self.al_iteration = al_iteration

        for i in range(epoch):
            self.train()
            self._reset_metrics("train")
            loader = DataLoader(
                dataset,
                batch_size,
                True,
                num_workers=workers,
                collate_fn=collate_fn,
                worker_init_fn=seed_worker,
            )

            val_loader = DataLoader(
                val_dataset,
                batch_size,
                False,
                num_workers=workers,
                collate_fn=None,
                worker_init_fn=seed_worker,
            )

            if verbose:
                loader = tqdm(loader)
            for data, target in loader:
                _ = self.train_on_batch(data, target, optimizers, use_cuda, regularizer)
                if verbose:
                    loader.set_description(f"Epoch [{i+1}/{epoch}]")
                    loader.set_postfix(
                        loss=self.metrics["train_loss"].value,
                        acc=self.metrics["train_{}".format(self.track_metric[0])].value,
                        val_loss=self.metrics["val_loss"].value,
                        val_acc=self.metrics[
                            "val_{}".format(self.track_metric[0])
                        ].value,
                    )

            for optimizer in optimizers:
                optimizer.zero_grad()  # Assert that the gradient is flushed.
            history.append(self.metrics["train_loss"].value)

            # Validation Loop
            if val_dataset is not None:
                val_loss = self.test_on_dataset(
                    val_dataset,
                    batch_size,
                    use_cuda,
                    workers=workers,
                    average_predictions=average_predictions,
                    validate=True,
                )

                # Learning rate scheduling
                if schedulers is not None:
                    for scheduler in schedulers:
                        if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                            scheduler.step(val_loss)
                        if isinstance(
                            scheduler,
                            (
                                optim.lr_scheduler.StepLR,
                                optim.lr_scheduler.LambdaLR,
                            ),
                        ):
                            scheduler.step()

                # Early stopping
                if val_loss <= best_loss:
                    best_loss = val_loss
                    patience_counter = 0
                    if return_best_weights:
                        best_weights = deepcopy(self.state_dict())
                else:
                    patience_counter += 1

                log_dict = {
                    "epoch": i + 1,
                    f"loss_{al_iteration}": self.metrics["train_loss"].value,
                    f"val_loss_{al_iteration}": self.metrics["val_loss"].value,
                    f"lr_{al_iteration}": optimizers[0].param_groups[0]["lr"],
                }

                # This is used to track the metrics, such as iou, accuracy, pac, pui, pavpu
                for m in self.track_metric:
                    if "train_{}".format(m) in self.metrics.keys():
                        train_m = self.metrics["train_{}".format(m)].value
                        if isinstance(train_m, dict):
                            train_df = wandb.Table(dataframe=pd.DataFrame(train_m))
                            log_dict["train_pac"] = train_df
                        else:
                            log_dict[f"{m}_{al_iteration}"] = train_m
                    if "val_{}".format(m) in self.metrics.keys():
                        val_m = self.metrics["val_{}".format(m)].value
                        if isinstance(val_m, dict):
                            val_df = wandb.Table(dataframe=pd.DataFrame(val_m))
                            log_dict["val_pac"] = val_df

                        else:
                            log_dict[f"val_{m}_{al_iteration}"] = val_m

                # log in wandb:
                wandb.log(log_dict)

                if patience_counter == patience:
                    if early_stopping:
                        break

        if isinstance(self.models[0], (MIR)):
            for model in self.models:
                return_true_labels = True if model.density_model == "knn" else False
                features_train, pred_train = extract_features(
                    model=model,
                    dataset=loader,
                    cuda=use_cuda,
                    return_true_labels=return_true_labels,
                    segmentation=(model.decoder is None),  # if decoder is none -> seg
                )

                features_val, pred_val = extract_features(
                    model=model,
                    dataset=val_loader,
                    cuda=use_cuda,
                    return_true_labels=return_true_labels,
                    segmentation=(model.decoder is None),  # if decoder is none -> seg
                )

                n_classes = self.models[0].nr_classes

                hist_train = np.array(
                    [np.sum(pred_train == c) for c in range(n_classes)]
                )
                plt.bar(list(map(str, range(n_classes))), hist_train)
                plt.yscale("log")
                plt.title("Training Class Distribution")
                addlabels(list(map(str, range(n_classes))), hist_train)
                fig = plt.gcf()
                img_train = fig2img(fig)
                plt.cla()

                hist_val = np.array([np.sum(pred_val == c) for c in range(n_classes)])
                plt.bar(list(map(str, range(n_classes))), hist_val)
                plt.yscale("log")
                plt.title("Validation Class Distribution")
                fig = plt.gcf()
                addlabels(list(map(str, range(n_classes))), hist_val)
                img_val = fig2img(fig)
                plt.cla()

                wandb.log(
                    {
                        f"train_histo_{self.al_iteration}": wandb.Image(
                            img_train,
                            caption="Training Class Distribution",
                        ),
                        f"val_histo_{self.al_iteration}": wandb.Image(
                            img_val,
                            caption="Validation Class Distribution",
                        ),
                    }
                )

                model.density.fit(
                    x=features_train, y=pred_train, x_val=features_val, y_val=pred_val
                )

        log.info("Training complete", train_loss=self.metrics["train_loss"].value)

        return history, best_weights

    def test_on_dataset(
        self,
        dataset: Dataset,
        batch_size: int,
        use_cuda: bool,
        workers: int = 4,
        collate_fn: Optional[Callable] = None,
        average_predictions: int = 1,
        validate: bool = False,
    ):
        """
        Test the model on a Dataset `dataset`.
        Args:
            dataset (Dataset): Dataset to evaluate on.
            batch_size (int): Batch size used for evaluation.
            use_cuda (bool): Use Cuda or not.
            workers (int): Number of workers to use.
            collate_fn (Optional[Callable]): The collate function to use.
            average_predictions (int): The number of predictions to average to
                compute the test loss.
            validate (bool): validation of data during training
        Returns:
            Average loss value over the dataset.
        """
        self.eval()

        if validate:
            self._reset_metrics("val")
        else:
            log.info("Starting evaluating", dataset=len(dataset))
            self._reset_metrics("test")

        visualize = True
        for data, target in DataLoader(
            dataset,
            batch_size,
            False,
            num_workers=workers,
            collate_fn=collate_fn,
            worker_init_fn=seed_worker,
        ):
            _ = self.test_on_batch(
                data,
                target,
                cuda=use_cuda,
                average_predictions=average_predictions,
                validate=validate,
                visualize=visualize,
            )
            visualize = False

        if validate:
            return self.metrics["val_loss"].value
        else:
            log.info("Testing complete", test_loss=self.metrics["test_loss"].value)
            return self.metrics["test_loss"].value

    def train_and_test_on_datasets(
        self,
        train_dataset: Dataset,
        test_dataset: Dataset,
        optimizer: Optimizer,
        batch_size: int,
        epoch: int,
        use_cuda: bool,
        workers: int = 4,
        collate_fn: Optional[Callable] = None,
        regularizer: Optional[Callable] = None,
        return_best_weights=False,
        patience=None,
        min_epoch_for_es=0,
    ):
        """
        Train and test the model on both Dataset `train_dataset`, `test_dataset`.
        Args:
            train_dataset (Dataset): Dataset to train on.
            test_dataset (Dataset): Dataset to evaluate on.
            optimizer (Optimizer): Optimizer to use during training.
            batch_size (int): Batch size used.
            epoch (int): Number of epoch to train on.
            use_cuda (bool): Use Cuda or not.
            workers (int): Number of workers to use.
            collate_fn (Optional[Callable]): The collate function to use.
            regularizer (Optional[Callable]): The loss regularization for training.
            return_best_weights (bool): If True, will keep the best weights and return them.
            patience (Optional[int]): If provided, will use early stopping to stop after
                                        `patience` epoch without improvement.
            min_epoch_for_es (int): Epoch at which the early stopping starts.
        Returns:
            History and best weights if required.
        """
        best_weight = None
        best_loss = 1e10
        best_epoch = 0
        hist = []
        for e in range(epoch):
            _ = self.train_on_dataset(
                train_dataset,
                optimizer,
                batch_size,
                1,
                use_cuda,
                workers,
                collate_fn,
                regularizer,
            )
            te_loss = self.test_on_dataset(
                test_dataset, batch_size, use_cuda, workers, collate_fn
            )
            hist.append({k: v.value for k, v in self.metrics.items()})
            if te_loss < best_loss:
                best_epoch = e
                best_loss = te_loss
                if return_best_weights:
                    best_weight = deepcopy(self.state_dict())

            if (
                patience is not None
                and (e - best_epoch) > patience
                and (e > min_epoch_for_es)
            ):
                # Early stopping
                break

        if return_best_weights:
            return hist, best_weight
        else:
            return hist

    def predict_on_dataset_generator(
        self,
        dataset: Dataset,
        batch_size: int,
        iterations: int,
        use_cuda: bool,
        workers: int = 4,
        collate_fn: Optional[Callable] = None,
        half=False,
        verbose=True,
    ):
        """
        Use the model to predict on a dataset `iterations` time.
        Args:
            dataset (Dataset): Dataset to predict on.
            batch_size (int):  Batch size to use during prediction.
            iterations (int): Number of iterations per sample.
            use_cuda (bool): Use CUDA or not.
            workers (int): Number of workers to use.
            collate_fn (Optional[Callable]): The collate function to use.
            half (bool): If True use half precision.
            verbose (bool): If True use tqdm to display progress
        Notes:
            The "batch" is made of `batch_size` * `iterations` samples.
        Returns:
            Generators [batch_size, n_classes, ..., n_iterations].
        """
        self.eval()
        if len(dataset) == 0:
            return None

        log.info("Start Predict", dataset=len(dataset))
        collate_fn = collate_fn or default_collate
        loader = DataLoader(
            dataset,
            batch_size,
            False,
            num_workers=workers,
            collate_fn=collate_fn,
            worker_init_fn=seed_worker,
        )
        if verbose:
            loader = tqdm(loader, total=len(loader), file=sys.stdout)

        visualize = True
        for idx, (data, _) in enumerate(loader):
            preds = []
            for model in self.models:
                pred = self.predict_on_batch(
                    model,
                    data,
                    iterations,
                    use_cuda,
                    pool_prediction=True,
                    visualize=visualize,
                )
                visualize = False
                preds.append(pred)

            if len(self.models) == 1:
                pred = preds[0]
            elif len(self.models) > 1 and preds[0].shape[2] == 1:
                pred = torch.stack(preds, dim=-1).squeeze(2)
            else:
                raise NotImplemented

            pred = map_on_tensor(lambda x: x.detach(), pred)
            if half:
                pred = map_on_tensor(lambda x: x.half(), pred)
            yield map_on_tensor(lambda x: x.cpu().numpy(), pred)

    def predict_on_dataset(
        self,
        dataset: Dataset,
        batch_size: int,
        iterations: int,
        use_cuda: bool,
        workers: int = 4,
        collate_fn: Optional[Callable] = None,
        half=False,
        verbose=True,
    ):
        """
        Use the model to predict on a dataset `iterations` time.
        Args:
            dataset (Dataset): Dataset to predict on.
            batch_size (int):  Batch size to use during prediction.
            iterations (int): Number of iterations per sample.
            use_cuda (bool): Use CUDA or not.
            workers (int): Number of workers to use.
            collate_fn (Optional[Callable]): The collate function to use.
            half (bool): If True use half precision.
            verbose (bool): If True use tqdm to show progress.
        Notes:
            The "batch" is made of `batch_size` * `iterations` samples.
        Returns:
            Array [n_samples, n_outputs, ..., n_iterations].
        """
        preds = list(
            self.predict_on_dataset_generator(
                dataset=dataset,
                batch_size=batch_size,
                iterations=iterations,
                use_cuda=use_cuda,
                workers=workers,
                collate_fn=collate_fn,
                half=half,
                verbose=verbose,
            )
        )
        if len(preds) > 0 and not isinstance(preds[0], Sequence):
            # Is an Array or a Tensor
            return np.vstack(preds)
        return [np.vstack(pr) for pr in zip(*preds)]

    def train_on_batch(
        self,
        data,
        target,
        optimizers,
        cuda=False,
        regularizer: Optional[Callable] = None,
    ):
        """
        Train the current model on a batch using `optimizer`.
        Args:
            data (Tensor): The model input.
            target (Tensor): The ground truth.
            optimizers (optim.Optimizer): list of optimizer.
            cuda (bool): Use CUDA or not.
            regularizer (Optional[Callable]): The loss regularization for training.
        Returns:
            Tensor, the loss computed from the criterion.
        """

        if cuda:
            data, target = to_cuda(data), to_cuda(target)

        losses = []
        outputs = []
        for model, optimizer in zip(self.models, optimizers):
            optimizer.zero_grad()
            output = model(data, return_reconstructions=True)

            if isinstance(model, MIR):
                loss = model.compute_loss(self.criterion, data, target, output)
                output = output["prediction"]

            elif isinstance(model, (BNN, BCNN)):
                # BayesNet implies additional KL-loss.
                loss = self.criterion(output, target)
                kl_loss = model.kl_div_weight * model.kl_loss()
                loss += kl_loss
            else:
                loss = self.criterion(output, target)

            if regularizer:
                regularized_loss = loss + regularizer()
                regularized_loss.backward()
            else:
                loss.backward()

            outputs.append(output)
            losses.append(loss)

            optimizer.step()

        # average the ensembles, if any
        output = torch.mean(torch.stack(outputs), dim=0)
        loss = torch.mean(torch.stack(losses), dim=0)

        self._update_metrics(output, target, loss, filter="train")
        return loss

    def test_on_batch(
        self,
        data: torch.Tensor,
        target: torch.Tensor,
        cuda: bool = False,
        average_predictions: int = 1,
        validate: bool = False,
        visualize=False,
    ):
        """
        Test the current model on a batch.
        Args:
            data (Tensor): The model input.
            target (Tensor): The ground truth.
            cuda (bool): Use CUDA or not.
            average_predictions (int): The number of predictions to average to
                compute the test loss.
            validate (bool): if the model is validating data during training
            visualize (bool): if to visualize in wandb
        Returns:
            Tensor, the loss computed from the criterion.
        """
        with torch.no_grad():
            if cuda:
                data, target = to_cuda(data), to_cuda(target)

            losses = []
            outputs = []
            for model in self.models:
                preds_it = self.predict_on_batch(
                    model,
                    data,
                    iterations=average_predictions,
                    cuda=cuda,
                )
                preds = preds_it.mean(-1)
                loss = self.criterion(preds, target)
                outputs.append(preds)
                losses.append(loss)

            # Check if either MCD iteration or Ensemble
            if len(self.models) == 1:
                pass
            elif len(self.models) > 1 and preds_it.shape[-1] == 1:
                preds_it = torch.stack(outputs, dim=-1)
            else:
                raise ValueError

            # average the ensembles, if any
            output = torch.mean(torch.stack(outputs, dim=-1), dim=-1)
            loss = torch.mean(torch.stack(losses), dim=0)

            if visualize and len(output.shape) > 2:
                phase = "val" if validate else "test"
                # Prediction
                batch_size = output.shape[0]
                batch_tensor = torch.max(output, 1)[1].long()
                batch_tensor = CITYSCAPE_PALETTE[batch_tensor].permute(0, 3, 1, 2)

                # Target
                batch_tensor_true = deepcopy(target)
                batch_tensor_true[batch_tensor_true == 255] = 19
                batch_tensor_true = CITYSCAPE_PALETTE[batch_tensor_true.long()].permute(
                    0, 3, 1, 2
                )
                img = torch.cat([batch_tensor, batch_tensor_true])
                grid_img = torchvision.utils.make_grid(img, nrow=batch_size)
                img = wandb.Image(
                    grid_img,
                    caption="Top: Predictions, Bottom: Ground Truth",
                )

                # log in wandb
                wandb.log({f"{phase}_vis_{self.al_iteration}": img})

            if validate:
                self._update_metrics(preds_it, target, loss, "val", reduce=True)
            else:
                if any(
                    [isinstance(v, (PAC, AUROC)) for v in self.metrics.values()]
                ) and isinstance(self.heuristic, Precomputed):
                    uncertainty = self.predict_on_batch(
                        model,
                        data,
                        average_predictions,
                        cuda,
                        pool_prediction=True,
                        visualize=False,
                    )

                    self._update_metrics(
                        preds_it,
                        target,
                        loss,
                        "test",
                        reduce=True,
                        uncertainty=uncertainty,
                    )
                else:
                    self._update_metrics(preds_it, target, loss, "test", reduce=True)
            return loss

    def predict_on_batch(
        self,
        model,
        data,
        iterations=1,
        cuda=False,
        pool_prediction=False,
        visualize=False,
    ):
        """
        Get the model's prediction on a batch.
        Args:
            model (Model): single pytorch model
            data (Tensor): The model input.
            iterations (int): Number of prediction to perform.
            cuda (bool): Use CUDA or not.
            pool_prediction (bool): if the models are currently predicting on the pool
            visualize (bool): visualize the uncertainty
        Returns:
            Tensor, the loss computed from the criterion.
                    shape = {batch_size, nclass, n_iteration}.
        Raises:
            Raises RuntimeError if CUDA rans out of memory during data replication.
        """
        with torch.no_grad():
            if cuda:
                data = to_cuda(data)
            if self.replicate_in_memory and not isinstance(model, (BNN, BCNN)):
                data = map_on_tensor(lambda d: stack_in_memory(d, iterations), data)
                try:
                    if pool_prediction and isinstance(self.heuristic, Precomputed):
                        out = model.uncertainty(data)
                        # if visualize:
                        #    plt.imsave("debug_uncertainty.png", out[0])
                        out = torch.Tensor(out)
                        if cuda:
                            out = to_cuda(out)
                        return out
                    else:
                        out = model(data)
                except RuntimeError as e:
                    raise RuntimeError(
                        """CUDA ran out of memory while BaaL tried to replicate data. See the exception above.
                    Use `replicate_in_memory=False` in order to reduce the memory requirements.
                    Note that there will be some speed trade-offs"""
                    ) from e
                out = map_on_tensor(
                    lambda o: o.view([iterations, -1, *o.size()[1:]]), out
                )
                out = map_on_tensor(
                    lambda o: o.permute(1, 2, *range(3, o.ndimension()), 0), out
                )
            else:
                out = []
                for _ in range(iterations):
                    if pool_prediction and isinstance(self.heuristic, Precomputed):
                        pred = model.uncertainty(data)
                        pred = torch.Tensor(pred)
                        if cuda:
                            pred = to_cuda(pred)
                    else:
                        pred = model(data)
                    out.append(pred)
                out = _stack_preds(out)
            return out

    def get_params(self):
        """
        Return the parameters to optimize.
        Returns:
            Config for parameters.
        """
        return [model.parameters() for model in self.models]

    def state_dict(self):
        """Get the state dict(s)."""
        return [model.state_dict() for model in self.models]

    def load_state_dict(self, state_dicts, strict=True):
        """Load the model with `state_dict`."""
        for model, state_dict in zip(self.models, state_dicts):
            model.load_state_dict(state_dict, strict=strict)

    def train(self):
        """Set the model in `train` mode."""
        for model in self.models:
            model.train()

    def eval(self):
        """Set the model in `eval mode`."""
        for model in self.models:
            model.eval()

    def reset_fcs(self):
        """Reset all torch.nn.Linear layers."""

        def reset(m):
            if isinstance(m, torch.nn.Linear):
                m.reset_parameters()

        self.model.apply(reset)

    def reset_all(self):
        """Reset all *resetable* layers."""

        def reset(m):
            for m in self.model.modules():
                getattr(m, "reset_parameters", lambda: None)()

        self.model.apply(reset)


def mc_inference(model, data, iterations, replicate_in_memory):
    if replicate_in_memory:
        input_shape = data.size()
        batch_size = input_shape[0]
        try:
            data = torch.stack([data] * iterations)
        except RuntimeError as e:
            raise RuntimeError(
                """CUDA ran out of memory while BaaL tried to replicate data. See the exception above.
            Use `replicate_in_memory=False` in order to reduce the memory requirements.
            Note that there will be some speed trade-offs"""
            ) from e
        data = data.view(batch_size * iterations, *input_shape[1:])
        try:
            out = model(data)
        except RuntimeError as e:
            raise RuntimeError(
                """CUDA ran out of memory while BaaL tried to replicate data. See the exception above.
            Use `replicate_in_memory=False` in order to reduce the memory requirements.
            Note that there will be some speed trade-offs"""
            ) from e
        out = map_on_tensor(
            lambda o: o.view([iterations, batch_size, *o.size()[1:]]), out
        )
        out = map_on_tensor(
            lambda o: o.permute(1, 2, *range(3, o.ndimension()), 0), out
        )
    else:
        out = [model(data) for _ in range(iterations)]
        if isinstance(out[0], Sequence):
            out = [torch.stack(ts, dim=-1) for ts in zip(*out)]
        else:
            out = torch.stack(out, dim=-1)
    return out
