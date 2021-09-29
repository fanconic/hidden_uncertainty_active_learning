import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class BayesianLayer(torch.nn.Module):
    """
    Module implementing a single Bayesian feedforward layer.
    The module performs Bayes-by-backprop, that is, mean-field
    variational inference. It keeps prior and posterior weights
    (and biases) and uses the reparameterization trick for sampling.
    """

    def __init__(self, input_dim, output_dim, bias=True, dropout=0.0):
        """Defines a Bayesian Layer, with distribution over its weights
        Args:
            input_dim: size of the input data
            output_dim: size of the output data
            bias (default True): Check if biases are being used
            dropout (default 0.0): Dropout probability
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = bias
        self.dropout = dropout
        assert self.dropout < 1 and self.dropout >= 0

        self.prior_mu = 0
        self.prior_sigma = 0.1
        self.prior_logsigma = np.log(self.prior_sigma)
        self.weight_mu = nn.Parameter(torch.Tensor(output_dim, input_dim))
        self.weight_logsigma = nn.Parameter(torch.Tensor(output_dim, input_dim))

        # Dropout
        if self.dropout != 0:
            self.dropout_tensor = torch.Tensor(output_dim, input_dim).fill_(
                1 - self.dropout
            )

        # Initialize the weights correctly
        stdv = 1.0 / np.sqrt(self.weight_mu.size(1))
        self.weight_mu.data.uniform_(-stdv, stdv)
        self.weight_logsigma.data.fill_(self.prior_logsigma)

        if self.use_bias:
            self.bias_mu = nn.Parameter(torch.Tensor(output_dim))
            self.bias_logsigma = nn.Parameter(torch.Tensor(output_dim))

            # Initialize the biases correctly
            self.bias_mu.data.uniform_(-stdv, stdv)
            self.bias_logsigma.data.fill_(self.prior_logsigma)

        else:
            self.register_parameter("bias_mu", None)
            self.register_parameter("bias_logsigma", None)

    def forward(self, inputs):
        """Forward pass through the BNN
        Args:
            inputs: input data
        returns:
            processed data
        """
        weight = self.weight_mu + torch.exp(self.weight_logsigma) * torch.randn_like(
            self.weight_logsigma
        )

        if self.use_bias:
            bias = self.bias_mu + torch.exp(self.bias_logsigma) * torch.randn_like(
                self.bias_logsigma
            )

        else:
            bias = None

        if self.dropout != 0:
            dropouts = torch.bernoulli(self.dropout_tensor)
            weight = dropouts * weight

        return F.linear(inputs, weight, bias)

    def kl_divergence(self):
        """Computes the KL divergence between the priors and posteriors for this layer.
        returns:
            sum of kl_divergence of the weights (and the biases)
        """
        kl_loss = self._kl_divergence(self.weight_mu, self.weight_logsigma)
        if self.use_bias:
            kl_loss += self._kl_divergence(self.bias_mu, self.bias_logsigma)
        return kl_loss

    def _kl_divergence(self, mu, logsigma):
        """Computes the KL divergence between one Gaussian posterior
        and the Gaussian prior.
        Args:
            mu: the mean of the distribution
            logsigma: the log of the variance of the distribution
        returns:
            the KL divergence between Gaussian Posterior and Prior (in closed form solution because of the gaussian distribution)
        """
        kl = (
            self.prior_logsigma
            - logsigma
            + (torch.exp(logsigma) ** 2 + (mu - self.prior_mu) ** 2)
            / (2 * np.exp(self.prior_logsigma) ** 2)
            - 0.5
        )
        return kl.mean()


class BayesNet(torch.nn.Module):
    """
    Module implementing a Bayesian feedforward neural network using
    BayesianLayer objects.
    """

    def __init__(
        self, input_size, output_size, num_layers, width, use_bias=True, dropout=0.0
    ):
        """Defines a Bayesian Neural Network, with a distribution over the weights
        Args:
            input_size: size of the input data
            output_size: size of the output data
            num_layers: number of hidden layers
            width: number of neurons per hidden layer
            use_bias (default True): check if biases are used or not
            dropout (default): dropout probability
        """
        super().__init__()
        self.dropout = dropout
        self.use_bias = use_bias
        assert self.dropout < 1 and self.dropout >= 0
        input_layer = torch.nn.Sequential(
            BayesianLayer(input_size, width, dropout=self.dropout, bias=self.use_bias),
            nn.ReLU(),
        )
        hidden_layers = [
            nn.Sequential(
                BayesianLayer(width, width, dropout=self.dropout, bias=self.use_bias),
                nn.ReLU(),
            )
            for _ in range(num_layers)
        ]
        output_layer = BayesianLayer(width, output_size, bias=self.use_bias)
        layers = [input_layer, *hidden_layers, output_layer]
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x):
        """Forward pass through the neural network
        Args:
            x: input data
        Returns:
            out: logit data
        """
        return self.net(x)

    def predict_class_probs(self, x, num_forward_passes=50):
        """Forward pass through the neural network and predicts class probability, via multiple forward passes
        Args:
            x: input data
            num_forward_passes: number of forward passes
        Returns:
            out: class probabilities
        """
        ys = []
        for _ in range(num_forward_passes):
            y_hat = self.net(x)
            softmax_layer = nn.Softmax(dim=1)
            y_hat = softmax_layer(y_hat)

            ys.append(y_hat.detach().numpy())

        ys = np.array(ys)
        probs = ys.mean(axis=0)
        return torch.Tensor(probs)

    def kl_loss(self):
        """Computes the KL divergence loss for all layers.
        returns:
            sum of kl divergence loss over all layers
        """
        kl_sum = 0

        for layer in self.net.modules():
            if isinstance(layer, BayesianLayer):
                kl_sum += layer.kl_divergence()
        return kl_sum
