# Taken from https://github.com/kumar-shridhar/PyTorch-BayesianCNN/blob/master/metrics.py

import numpy as np
import torch.nn.functional as F
from torch import nn
import torch


def ece(probs, labels, n_bins=30):
    """
    probs has shape [n_examples, n_classes], labels has shape [n_class] -> np.float
    Computes the Expected Calibration Error (ECE). Many options are possible,
    in this implementation, we provide a simple version.

    Using a uniform binning scheme on the full range of probabilities, zero
    to one, we bin the probabilities of the predicted label only (ignoring
    all other probabilities). For the ith bin, we compute the avg predicted
    probability, p_i, and the bin's total accuracy, a_i. We then compute the
    ith calibration error of the bin, |p_i - a_i|. The final returned value
    is the weighted average of calibration errors of each bin.

    args:
        probs: probabilities of predictions
        labels: true labels of the predictions
        n_bins: number of bins for calculation
    retuns:
        expected calibration error
    """
    n_examples, n_classes = probs.shape

    # assume that the prediction is the class with the highest prob.
    preds = np.argmax(probs, axis=1)

    onehot_labels = np.eye(n_classes)[labels]

    predicted_class_probs = probs[range(n_examples), preds]

    # Use uniform bins on the range of probabilities, i.e. closed interval [0.,1.]
    bin_upper_edges = np.histogram_bin_edges([], bins=n_bins, range=(0.0, 1.0))
    bin_upper_edges = bin_upper_edges[1:]  # bin_upper_edges[0] = 0.

    probs_as_bin_num = np.digitize(predicted_class_probs, bin_upper_edges)
    sums_per_bin = np.bincount(
        probs_as_bin_num, minlength=n_bins, weights=predicted_class_probs
    )
    sums_per_bin = sums_per_bin.astype(np.float32)

    total_per_bin = (
        np.bincount(probs_as_bin_num, minlength=n_bins)
        + np.finfo(sums_per_bin.dtype).eps
    )  # division by zero
    avg_prob_per_bin = sums_per_bin / total_per_bin

    accuracies = onehot_labels[range(n_examples), preds]  # accuracies[i] is 0 or 1
    accuracies_per_bin = (
        np.bincount(probs_as_bin_num, weights=accuracies, minlength=n_bins)
        / total_per_bin
    )

    prob_of_being_in_a_bin = total_per_bin / float(n_examples)

    ece_ret = np.abs(accuracies_per_bin - avg_prob_per_bin) * prob_of_being_in_a_bin
    ece_ret = np.sum(ece_ret)
    return ece_ret


class ELBO(nn.Module):
    def __init__(self, train_size):
        super(ELBO, self).__init__()
        self.train_size = train_size

    def forward(self, input, target, kl, beta):
        assert not target.requires_grad
        return F.nll_loss(input, target, reduction="mean") * self.train_size + beta * kl


def acc(outputs, targets):
    return np.mean(outputs.cpu().numpy().argmax(axis=1) == targets.data.cpu().numpy())


def calculate_kl(mu_q, sig_q, mu_p, sig_p):
    """Computes the KL divergence between one Gaussian posterior
    and the Gaussian prior.
    Args:
        mu_q: the mean of the distribution
        sig_q: the log of the variance of the distribution
        mu_p: mean of the prior
        sig_p: variance of the prior
    returns:
        the KL divergence between Gaussian Posterior and Prior
        (in closed form solution because of the gaussian distribution)
    """
    kl = (
        0.5
        * (
            2 * torch.log(sig_p / sig_q)
            - 1
            + (sig_q / sig_p).pow(2)
            + ((mu_p - mu_q) / sig_p).pow(2)
        ).sum()
    )
    return kl


def get_beta(batch_idx, m, beta_type, epoch, num_epochs):
    if type(beta_type) is float:
        return beta_type

    if beta_type == "Blundell":
        beta = 2 ** (m - (batch_idx + 1)) / (2 ** m - 1)
    elif beta_type == "Soenderby":
        if epoch is None or num_epochs is None:
            raise ValueError(
                "Soenderby method requires both epoch and num_epochs to be passed."
            )
        beta = min(epoch / (num_epochs // 4), 1)
    elif beta_type == "Standard":
        beta = 1 / m
    else:
        beta = 0
    return beta
