# Lint as: python3
# Taken from: Janis Postels Hidden Unvertainty AAAI
"""Class-conditional density models."""

from typing import Any
import numpy as np
import scipy
import sklearn.mixture as mixture
import sklearn.decomposition as decomposition
import sklearn.preprocessing as preprocessing
import scipy.ndimage
from copy import deepcopy
from sklearn.neighbors import KNeighborsClassifier
from imblearn.under_sampling import RandomUnderSampler


def metric_function(
    log_prob_train, log_prob_val, dimension, metric="L2", reduction="mean"
):
    """Calulcates a metric between the log probability of the training set and the log probability of the validation set
    Args:
        log_prob_train: log probability of the training examples
        log_prob_val: log probability of the validation examples
        dimension: dimension of the samples
        metric (default L2): metric
        reduction (default "mean"): reduction method
    Returns:
        metric
    """
    if reduction == "mean":
        log_prob_train = log_prob_train.mean()
        log_prob_val = log_prob_val.mean()

    elif reduction == "sum":
        log_prob_train = log_prob_train.sum()
        log_prob_val = log_prob_val.sum()

    elif reduction == "max":
        log_prob_train = log_prob_train.max()
        log_prob_val = log_prob_val.max()

    elif reduction == "min":
        log_prob_train = log_prob_train.min()
        log_prob_val = log_prob_val.min()

    elif reduction == "median":
        log_prob_train = np.median(log_prob_train)
        log_prob_val = np.median(log_prob_val)

    else:
        raise NotImplementedError

    if metric == "L2":
        return ((log_prob_train - log_prob_val) / dimension) ** 2
    elif metric == "L1":
        return np.abs(log_prob_train / dimension - log_prob_val / dimension)
    elif metric == "train":
        return -log_prob_train / dimension
    elif metric == "val":
        return -log_prob_val / dimension
    elif metric == "train_l2":
        return (
            -log_prob_train / dimension
            + ((log_prob_train - log_prob_val) / dimension) ** 2
        )
    elif metric == "train_val_l2":
        return (
            -log_prob_train / dimension
            - log_prob_val / dimension
            + ((log_prob_train - log_prob_val) / dimension) ** 2
        )
    elif metric == "val_l2":
        return (
            -log_prob_val / dimension
            + ((log_prob_train - log_prob_val) / dimension) ** 2
        )
    elif metric == "train_val_l1":
        return (
            -log_prob_train / dimension
            - log_prob_val / dimension
            + np.abs(log_prob_train / dimension - log_prob_val / dimension)
        )
    elif metric == "val_l1":
        return -log_prob_val / dimension + np.abs(
            log_prob_train / dimension - log_prob_val / dimension
        )
    else:
        raise NotImplementedError


def class_probs(labels, num_classes):
    """Calculates the probability of each class
    Args:
        labels (np.array): the labels
        num_classes (int): number of classes
    returns:
        np array of length num_classes with their respective probability
    """
    class_n = len(labels)
    class_count = np.array([np.sum(labels == c) for c in range(num_classes)])
    class_prob = class_count / class_n
    return class_prob


class ClassConditionalGMM(object):
    """Wraps conditional densities."""

    def __init__(
        self,
        n_components: int = 1,
        nr_classes: int = 10,
        red_dim: int = 64,
        normalize_features: bool = True,
        greedy_search: bool = False,
        search_step_size: int = 10,
        metric: str = "L2",
        reduction: str = "mean",
    ):
        super(ClassConditionalGMM, self).__init__()
        self.n_components = n_components
        self.nr_classes = nr_classes
        self.normalize_features = normalize_features
        self.class_conditional_densities = []
        self.greedy_search = greedy_search
        self.search_step_size = search_step_size
        self.reduce = red_dim
        self.metric = metric
        self.reduction = reduction
        self.class_distribution = None

        if red_dim != -1:
            self.pca = decomposition.PCA(n_components=red_dim)
        else:
            self.pca = None
        for i in range(self.nr_classes):
            self.class_conditional_densities.append(
                mixture.GaussianMixture(
                    n_components=self.n_components,
                    covariance_type="full",
                    random_state=42,
                )
            )

    def fit(
        self,
        x: Any,
        y: Any,
        x_val: Any,
        y_val: Any,
    ):
        """Fit output-conditional density. The distribution of x associated with each class is estimated using separate GMM.

        Args:
            x: Training data
            y: Predictions on training data
            x_val: Validation data
            y_val: Predictions on validation data

        Returns:
            List of GMMs
        """
        nr_samples = x.shape[0]
        print(nr_samples)

        # calculate the class distribution to figure out the densities
        self.class_distribution = class_probs(y, self.nr_classes)

        if self.normalize_features:
            x = preprocessing.normalize(x)
            x_val = preprocessing.normalize(x_val)

        best_dim = x.shape[1]
        max_dim = min(x.shape[0], x.shape[1]) if self.greedy_search else x.shape[1]
        red_dim = self.search_step_size if self.greedy_search else max_dim
        min_diff = np.inf
        best_model = None
        best_pca = None

        while red_dim <= max_dim:
            if self.greedy_search and max_dim != red_dim:
                self.pca = decomposition.PCA(n_components=red_dim)
            else:
                if self.reduce:
                    self.pca = None
            if self.pca:
                print("Fitting PCA...", flush=True)
                self.pca.fit(x)
                x_in = self.pca.transform(x)
                x_val_in = self.pca.transform(x_val)
            else:
                x_in = x
                x_val_in = x_val

            diffs = []
            for i in range(self.nr_classes):
                if np.sum(y == i) > 1:  # sanity check whether this idx exists
                    self.class_conditional_densities[
                        i
                    ] = self.class_conditional_densities[i].fit(X=x_in[y == i])

                    # log log_prob on train/val
                    if np.sum(y_val == i) > 0:
                        log_prob_train = self.class_conditional_densities[
                            i
                        ].score_samples(x_in[y == i])

                        log_prob_val = self.class_conditional_densities[
                            i
                        ].score_samples(x_val_in[y_val == i])
                        if not self.greedy_search:
                            print(
                                f"{i}-th component log probs | Train: {log_prob_train.mean()} | Val: {log_prob_val.mean()}"
                            )
                        else:
                            print(
                                f"{i}-th component log probs | Train: {(log_prob_train/red_dim).mean()} | Val: {(log_prob_val/red_dim).mean()}"
                            )
                        diff = metric_function(
                            log_prob_train,
                            log_prob_val,
                            red_dim,
                            metric=self.metric,
                            reduction=self.reduction,
                        )

                        diffs.append(diff)

            # compute the average negative log likelihood
            avg_diff = np.mean(diffs)

            if self.greedy_search:
                print(
                    f"KEYWORD NR_SAMPLES:{nr_samples} DIM:{red_dim} AVG_DIFF:{avg_diff:.4f}"
                )

            # update model and params if we find a better log likelihood
            if avg_diff < min_diff:
                min_diff = avg_diff
                best_model = deepcopy(self.class_conditional_densities)
                if self.greedy_search:
                    best_pca = deepcopy(self.pca)
                    best_dim = red_dim

            # Increase the dimensions
            if red_dim == max_dim:
                break
            else:
                red_dim = min(red_dim + self.search_step_size, max_dim)

        self.class_conditional_densities = best_model
        self.pca = best_pca
        if self.greedy_search:
            print(
                "best average differences at {} dimensions: {}".format(
                    best_dim, min_diff
                )
            )

    def class_conditional_log_probs(self, x: Any) -> Any:
        class_idx = []
        log_probs = []
        if self.normalize_features:
            x = preprocessing.normalize(x)
        if self.pca:
            x = self.pca.transform(x)
        for density in self.class_conditional_densities:
            try:
                log_probs.append(np.expand_dims(density.score_samples(x), -1))
                class_idx.append(True)
            except:
                class_idx.append(False)
        return np.concatenate(log_probs, -1), class_idx

    def marginal_log_probs(self, x: Any):
        """Computes marginal likelihood (epistemic uncertainty)of x.
        Args:
            x (np.array): array of dim batch_size x features
        Returns:
          epistemic uncertainty: dim batch_size
        """
        cc_log_probs, class_idx = self.class_conditional_log_probs(x)
        return scipy.special.logsumexp(
            cc_log_probs, b=self.class_distribution[class_idx], axis=-1
        )


class KNearestNeighbour(object):
    """Wraps conditional densities."""

    def __init__(
        self,
        n_neigbours: int = 5,
        red_dim: int = 64,
        normalize_features: bool = True,
        weights="uniform",
        metric="euclidean",
        nr_classes=10,
        max_samples=5000,
    ):
        super(KNearestNeighbour, self).__init__()
        self.normalize_features = normalize_features
        self.nr_classes = nr_classes
        self.max_samples = max_samples
        self.class_distribution = None

        if red_dim != -1:
            self.pca = decomposition.PCA(n_components=red_dim)
        else:
            self.pca = None

        self.knn = KNeighborsClassifier(
            metric=metric,
            n_neighbors=n_neigbours,
            weights=weights,
            n_jobs=-1,
            algorithm="ball_tree",
        )

    def fit(
        self,
        x: Any,
        y: Any,
        x_val: Any,
        y_val: Any,
    ):
        """Fit KNN.
        Args:
            x: Training data
            y: Predictions on training data
            x_val: Validation data
            y_val: Predictions on validation data
        Returns:
            fitted KNN
        """
        if self.normalize_features:
            x = preprocessing.normalize(x)
            x_val = preprocessing.normalize(x_val)

        self.class_distribution = class_probs(y, self.nr_classes)

        if self.max_samples != -1:
            class_n = np.array([np.sum(y == c) for c in range(self.nr_classes)])
            class_n_val = np.array([np.sum(y_val == c) for c in range(self.nr_classes)])

            y_samples = {}
            y_samples_val = {}
            for c in range(self.nr_classes):
                if class_n[c] != 0:
                    y_samples[c] = min(class_n[c], self.max_samples)
                    y_samples_val[c] = min(class_n_val[c], self.max_samples)

            rus = RandomUnderSampler(sampling_strategy=y_samples, random_state=42)
            rus_val = RandomUnderSampler(
                sampling_strategy=y_samples_val, random_state=42
            )
            x, y = rus.fit_resample(x, y)
            x_val, y_val = rus_val.fit_resample(x_val, y_val)

        print("Fitting KNN")
        self.knn.fit(x, y)
        self.y = y
        eval = self.evaluate(x_val)
        print(f"Mean distance validation: {eval.mean():.4f}")

    def evaluate(self, x):
        y_preds = self.knn.predict(x)
        distances, indices = self.knn.kneighbors(x)

        relevant_distances = []
        for y_pred, distance, index in zip(y_preds, distances, indices):
            relevant_distances.append(np.nanmean(distance[self.y[index] == y_pred]))
            # relevant_distances.append(distance.mean())

        return np.array(relevant_distances)

    def class_conditional_log_probs(self, x: Any) -> Any:
        if self.normalize_features:
            x = preprocessing.normalize(x)
        if self.pca:
            x = self.pca.transform(x)
        return self.evaluate(x)

    def marginal_log_probs(self, x: Any):
        """Computes the distances as proxy of the density (kernel density approximation).
        Args:
            x (np.array): array of dim batch_size x features
        Returns:
          distance: dim batch_size
        """
        return -1 * self.class_conditional_log_probs(x)


class ClassConditionalGMM_Seg(object):
    """Wraps conditional densities for segmenetation."""

    def __init__(
        self,
        n_components: int = 1,
        nr_classes: int = 10,
        red_dim: int = 64,
        normalize_features: bool = True,
        greedy_search: bool = False,
        search_step_size: int = 10,
        metric: str = "L2",
        reduction: str = "mean",
    ):
        super(ClassConditionalGMM_Seg, self).__init__()
        self.n_components = n_components
        self.nr_classes = nr_classes
        self.normalize_features = normalize_features
        self.class_conditional_densities = []
        self.greedy_search = greedy_search
        self.search_step_size = search_step_size
        self.reduce = red_dim
        self.metric = metric
        self.reduction = reduction
        self.class_distribution = None

        if red_dim != -1:
            self.pca = decomposition.PCA(n_components=red_dim)
        else:
            self.pca = None
        for i in range(self.nr_classes):
            self.class_conditional_densities.append(
                mixture.GaussianMixture(
                    n_components=self.n_components,
                    covariance_type="full",
                    random_state=42,
                )
            )

    def fit(
        self,
        x: Any,
        y: Any,
        x_val: Any,
        y_val: Any,
    ):
        """Fit output-conditional density. The distribution of x associated with each class is estimated using separate GMM.

        Args:
            x: Training data
            y: Predictions on training data
            x_val: Validation data
            y_val: Predictions on validation data

        Returns:
            List of GMMs
        """
        nr_samples = x.shape[0]
        print(nr_samples)

        # calculate the class distribution to figure out the densities
        self.class_distribution = class_probs(y, self.nr_classes)

        self.pcas = [None for i in range(self.nr_classes)]
        for i in range(self.nr_classes):
            best_dim = x.shape[1]
            max_dim = min(x.shape[0], x.shape[1]) if self.greedy_search else x.shape[1]
            red_dim = self.search_step_size if self.greedy_search else max_dim
            min_diff = np.inf
            best_model = None
            best_pca = None
            if np.sum(y == i) > 1:  # sanity check whether this idx exists
                while red_dim <= max_dim:
                    x_in = x[y == i]
                    if self.greedy_search and max_dim != red_dim:
                        pca = decomposition.PCA(n_components=red_dim)
                    else:
                        if self.reduce:
                            pca = None
                    if pca:
                        if red_dim > x_in.shape[0]:
                            break
                        pca.fit(x_in)
                        x_in = pca.transform(x_in)
                    if self.normalize_features:
                        x_in = preprocessing.normalize(x_in)
                    self.class_conditional_densities[
                        i
                    ] = self.class_conditional_densities[i].fit(X=x_in)

                    # log log_prob on train/val
                    if np.sum(y_val == i) > 0:
                        x_val_in = x_val[y_val == i]
                        if pca:
                            x_val_in = pca.transform(x_val_in)
                        if self.normalize_features:
                            x_val_in = preprocessing.normalize(x_val_in)
                        log_prob_train = self.class_conditional_densities[
                            i
                        ].score_samples(x_in)
                        log_prob_val = self.class_conditional_densities[
                            i
                        ].score_samples(x_val_in)
                        diff = metric_function(
                            log_prob_train,
                            log_prob_val,
                            red_dim,
                            metric=self.metric,
                            reduction=self.reduction,
                        )

                        # update model and params if we find a better log likelihood
                        if diff < min_diff:
                            min_diff = diff
                            best_model = deepcopy(self.class_conditional_densities[i])
                            best_pca = deepcopy(pca)
                            best_dim = red_dim

                    # Increase the dimensions
                    if red_dim == max_dim:
                        break
                    else:
                        red_dim = min(red_dim + self.search_step_size, max_dim)

                self.class_conditional_densities[i] = (
                    best_model
                    if best_model is not None
                    else self.class_conditional_densities[i]
                )
                self.pcas[i] = best_pca
                if self.greedy_search:
                    print(
                        "best average differences for {}-th component at {} dimensions: {}".format(
                            i, best_dim, min_diff
                        )
                    )

    def class_conditional_log_probs(self, x: Any) -> Any:
        class_idx = []
        log_probs = []
        for pca, density in zip(self.pcas, self.class_conditional_densities):
            try:
                x_in = x
                if pca is not None:
                    x_in = pca.transform(x_in)
                if self.normalize_features:
                    x_in = preprocessing.normalize(x_in)
                log_probs.append(np.expand_dims(density.score_samples(x_in), -1))
                class_idx.append(True)
            except:
                class_idx.append(False)
        return np.concatenate(log_probs, -1), class_idx

    def marginal_log_probs(self, x: Any):
        """Computes marginal likelihood (epistemic uncertainty)of x.
        Args:
            x (np.array): array of dim batch_size x features
        Returns:
          epistemic uncertainty: dim batch_size
        """
        cc_log_probs, class_idx = self.class_conditional_log_probs(x)
        return scipy.special.logsumexp(
            cc_log_probs, b=self.class_distribution[class_idx], axis=-1
        )
