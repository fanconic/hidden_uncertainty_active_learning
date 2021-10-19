import torch
from torch import nn
from src.models.MLP import MLP, MLPDecoder
from src.models.density_models import ClassConditionalGMM
from typing import Tuple, Dict, List
import numpy as np


class MIR(nn.Module):
    def __init__(self, model_configs):
        super().__init__()

        mir_configs = model_configs["mir_configs"]
        self.reconstruction_weight = mir_configs["reconstruction_weight"]
        self.warmup = mir_configs["warmup"]
        self.warmup_counter = self.warmup
        self.feature_dims = mir_configs["feature_dims"]
        self.backbone = mir_configs["backbone"]
        self.density_model = mir_configs["density_model"]
        self.normalize_features = mir_configs["normalize_features"]
        self.nr_classes = model_configs["output_size"]

        # encoder, decoder & loss function
        self.encoder = self._get_encoder(model_configs)
        self.decoder = self._get_decoder(model_configs)
        self.loss_func_reconstruction = nn.MSELoss()

        self.density = None
        self.init_density(self.normalize_features)

    def init_density(self, normalize_features: bool = True):
        # density model
        if self.density_model == "gmm":
            self.density = ClassConditionalGMM(
                nr_classes=self.nr_classes,
                red_dim=-1,
                normalize_features=normalize_features,
            )
        else:
            raise ValueError(f"Unknown density model {self.density_model}!")

    def _reconstruction_weight_to_loss_weights(
        self, use_warmup: bool = True
    ) -> Tuple[float, float]:
        if self.reconstruction_weight >= 1.0:
            w1, w2 = 1 / self.reconstruction_weight, 1.0
        else:
            w1, w2 = 1.0, self.reconstruction_weight
        if not use_warmup:
            return w1, w2
        if not self.warmup is None and self.warmup_counter > 0:
            w1 = w1 * (self.warmup - self.warmup_counter) / self.warmup
        return w1, w2

    def _get_encoder(self, model_configs: Dict) -> nn.Module:
        """Get the model encoder part
        Args:
            model_configs (Dict): Dict cintaining the necessary model information
        returns:
            torch.nn.Module
        """
        if self.backbone == "MLP":
            return MLP(model_configs)
        elif self.backbone == "resnet":
            NotImplementedError
        else:
            raise ValueError(f"Unknown backbone {self.backbone}!")

    def _get_decoder(self, model_configs: Dict) -> nn.Module:
        """Get the model decoder part
        Args:
            model_configs (Dict): Dict cintaining the necessary model information
        returns:
            torch.nn.Module
        """
        if self.backbone == "MLP":
            return MLPDecoder(model_configs)
        elif self.backbone == "resnet":
            NotImplementedError
        else:
            raise ValueError(f"Unknown backbone {self.backbone}!")

    def forward(
        self,
        inputs: torch.Tensor,
        return_reconstructions: bool = False,
        return_features: bool = False,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through the MIR network. If we have reconstruction loss, a dict
        is forwarded including the reconstruction and output features.
        Args:
            inputs (torch.Tensor): features
            return_reconstructions (bool): returns the reconstructions in a dict
            return_features (bool): return the features in a dict
        Returns:
            dict consisting of the output features and potentially the reconstructions
        """
        outputs, output_features = self.encoder(inputs, return_features=True)

        output_dict = {}
        output_dict["prediction"] = outputs

        if return_features:
            output_dict["features"] = output_features

        if return_reconstructions:
            if self.reconstruction_weight > 0.0:
                output_dict["reconstructions"] = self.decoder(output_features)

        if not return_reconstructions and not return_features:
            return outputs  # in case of a normal forward pass
        else:
            return output_dict

    def compute_loss(
        self,
        loss,
        x: torch.Tensor,
        y: torch.Tensor,
        out_dict: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Compute the loss function of the MIR
        Args:
            loss (nn.Criterion): loss function for the task at hand
            x: features
            y: labels
            out_dict: dict of predicted model
        Returns:
            output dictionary
        """
        (
            prediction_weight,
            reconstruction_weight,
        ) = self._reconstruction_weight_to_loss_weights()

        out_dict["loss_prediction"] = loss(out_dict["prediction"], y)
        out_dict["loss_weighted"] = prediction_weight * out_dict["loss_prediction"]
        if self.reconstruction_weight > 0.0:
            out_dict["loss_reconstruction"] = self.loss_func_reconstruction(
                x, out_dict["reconstructions"]
            )
            out_dict["loss_weighted"] += (
                reconstruction_weight * out_dict["loss_reconstruction"]
            )

        (
            prediction_weight,
            reconstruction_weight,
        ) = self._reconstruction_weight_to_loss_weights(use_warmup=False)
        if self.reconstruction_weight > 0.0:
            out_dict["loss"] = (
                prediction_weight * out_dict["loss_prediction"]
                + reconstruction_weight * out_dict["loss_reconstruction"]
            )
        else:
            out_dict["loss"] = out_dict["loss_prediction"]

        return out_dict["loss"]

    def predict_class_probs(self, data: Tuple[torch.Tensor]) -> torch.Tensor:
        """Computes class probabilities estimates given x.

        Args:
          data: batch

        Returns:
          dictionary with entries 'prediction' and 'uncertainty'
        """
        out = self.forward(data)
        softmax = nn.Softmax(dim=1)
        out_preds = softmax(out)
        return out_preds

    def uncertainty(self, data: Tuple[torch.Tensor]) -> torch.Tensor:
        """Computes uncertainty estimates given x.

        Args:
          data: batch

        Returns:
          dictionary with entries 'prediction' and 'uncertainty'
        """
        output_dict = self.forward(inputs=data, return_features=True)
        uncertainty = -1 * self.density.marginal_log_probs(
            output_dict["features"].cpu().detach()
        )
        return np.expand_dims(uncertainty, axis=-1)
