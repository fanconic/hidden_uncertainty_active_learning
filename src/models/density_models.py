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


class ClassConditionalGMM(object):
    """Wraps conditional densities."""

    def __init__(
        self,
        n_components: int = 1,
        nr_classes: int = 10,
        red_dim: int = 64,
        normalize_features: bool = True,
    ):
        super(ClassConditionalGMM, self).__init__()
        self.n_components = n_components
        self.nr_classes = nr_classes
        self.normalize_features = normalize_features
        self.class_conditional_densities = []
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

    def fit(self, x: Any, y: Any, x_val: Any, y_val: Any):
        """Fit output-conditional density. The distribution of x associated with each class is estimated using separate GMM.

        Args:
          x: Training data
          y: Predictions on training data
          x_val: Validation data
          y_val: Predictions on validation data
          n_components: Number components in each GMM
          nr_classes:

        Returns:
          List of GMMs
        """

        if self.normalize_features:
            x = preprocessing.normalize(x)
            x_val = preprocessing.normalize(x_val)

        if self.pca:
            print("Fitting PCA...", flush=True)
            self.pca.fit(x)
            x = self.pca.transform(x)
            x_val = self.pca.transform(x_val)

        for i in range(self.nr_classes):
            if np.sum(y == i) > 1:  # sanity check whether this idx exists
                self.class_conditional_densities[i] = self.class_conditional_densities[
                    i
                ].fit(X=x[y == i])

                # log log_prob on train/val
                if np.sum(y_val == i) > 0:
                    log_prob_train = np.mean(
                        self.class_conditional_densities[i].score_samples(x[y == i])
                    )
                    log_prob_val = np.mean(
                        self.class_conditional_densities[i].score_samples(
                            x_val[y_val == i]
                        )
                    )
                    print(
                        f"{i}-th component log probs | Train: {log_prob_train} | Val: {log_prob_val}"
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
