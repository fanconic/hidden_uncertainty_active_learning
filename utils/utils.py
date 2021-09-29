import numpy as np
import torch
import os


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


def load_mnist():
    """Load the MNIST dataset"""

    mnist_path = "/data/rotated_mnist.npz"
    if not os.path.isfile(mnist_path):
        mnist_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "data/rotated_mnist.npz"
        )

    data = np.load(mnist_path)

    x_train = torch.from_numpy(data["x_train"]).reshape([-1, 784])
    y_train = torch.from_numpy(data["y_train"])

    dataset_train = torch.utils.data.TensorDataset(x_train, y_train)

    return dataset_train
