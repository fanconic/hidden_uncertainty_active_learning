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
    ):
        super(ClassConditionalGMM, self).__init__()
        self.n_components = n_components
        self.nr_classes = nr_classes
        self.normalize_features = normalize_features
        self.class_conditional_densities = []
        self.greedy_search = greedy_search
        self.search_step_size = search_step_size
        self.reduce = red_dim

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
        if nr_samples >= 10000:
            self.greedy_search = False
            print("Nr of samples: {}".format(nr_samples))

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
            if self.greedy_search:
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
                        log_prob_train = np.mean(
                            self.class_conditional_densities[i].score_samples(
                                x_in[y == i]
                            )
                        )
                        log_prob_val = np.mean(
                            self.class_conditional_densities[i].score_samples(
                                x_val_in[y_val == i]
                            )
                        )
                        if not self.greedy_search:
                            print(
                                f"{i}-th component log probs | Train: {log_prob_train} | Val: {log_prob_val}"
                            )
                        diffs.append(((log_prob_train - log_prob_val) / red_dim) ** 2)

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
        log_probs = []

        if self.normalize_features:
            x = preprocessing.normalize(x)
        if self.pca:
            x = self.pca.transform(x)
        for density in self.class_conditional_densities:
            try:
                log_probs.append(np.expand_dims(density.score_samples(x), -1))
            except:
                pass
        return np.concatenate(log_probs, -1)

    def class_conditional_probs(self, x: Any) -> Any:
        probs = []
        if self.normalize_features:
            x = preprocessing.normalize(x)
        if self.pca:
            x = self.pca.transform(x)
        for density in self.class_conditional_densities:
            try:
                probs.append(density.predict_proba(x))
            except:
                pass
        return np.concatenate(probs, -1)

    def marginal_log_probs(self, x: Any):
        """Computes marginal likelihood (epistemic uncertainty)of x. Assuming class balance.

        Args:
            x (np.array): array of dim batch_size x features

        Returns:
          epistemic uncertainty: dim batch_size
        """
        cc_log_probs = self.class_conditional_log_probs(x)
        return scipy.special.logsumexp(cc_log_probs, axis=-1)


class KNearestNeighbour(object):
    """Wraps conditional densities."""

    def __init__(
        self,
        n_neigbours: int = 5,
        red_dim: int = 64,
        normalize_features: bool = True,
        weights="uniform",
        metric="euclidean",
    ):
        super(KNearestNeighbour, self).__init__()
        self.normalize_features = normalize_features

        if red_dim != -1:
            self.pca = decomposition.PCA(n_components=red_dim)
        else:
            self.pca = None

        self.knn = KNeighborsClassifier(
            metric=metric,
            n_neighbors=n_neigbours,
            weights=weights,
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

        print("Fitting KNN")
        self.knn.fit(x, y)
        eval = self.evaluate(x_val)
        print(f"Mean distance validation: {eval.mean():.4f}")

    def evaluate(self, x):
        y_preds = self.knn.predict(x)
        distances, indices = self.knn.kneighbors(x)

        relevant_distances = []
        for y_pred, distance, index in zip(y_preds, distances, indices):
            relevant_distances.append(distance[self.knn._y[index] == y_pred].mean())
            # relevant_distances.append(distance.mean())

        return np.array(relevant_distances)

    def class_conditional_log_probs(self, x: Any) -> Any:
        if self.normalize_features:
            x = preprocessing.normalize(x)
        if self.pca:
            x = self.pca.transform(x)

        return self.evaluate(x)

    def marginal_log_probs(self, x: Any):
        """Computes marginal likelihood (epistemic uncertainty)of x. Assuming class balance.

        Args:
            x (np.array): array of dim batch_size x features

        Returns:
          epistemic uncertainty: dim batch_size
        """
        return -1 * self.class_conditional_log_probs(x)
