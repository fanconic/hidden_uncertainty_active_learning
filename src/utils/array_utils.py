# Taken from: https://github.com/ElementAI/baal/blob/a9cc0034c40d0541234a3c27ff5ccbd97278bcb3/baal/utils/array_utils.py#L27

import numpy as np
import torch
from scipy.special import softmax, expit


def to_prob(probabilities: np.ndarray):
    """
    If the probabilities array is not a distrubution will softmax it.
    Args:
        probabilities (array): [batch_size, num_classes, ...]
    Returns:
        Same as probabilities.
    """
    not_bounded = np.min(probabilities) < 0 or np.max(probabilities) > 1.0
    multiclass = probabilities.shape[1] > 1
    sum_to_one = np.allclose(probabilities.sum(1), 1)
    if not_bounded or (multiclass and not sum_to_one):
        if multiclass:
            probabilities = softmax(probabilities, 1)
        else:
            probabilities = expit(probabilities)
    return probabilities


def stack_in_memory(data, iterations):
    """
    Stack `data` `iterations` times on the batch axis.
    Args:
        data (Tensor): Data to stack
        iterations (int): Number of time to stack.
    Raises:
        RuntimeError when CUDA is out of memory.
    Returns:
        Tensor with shape [batch_size * iterations, ...]
    """
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
    return data


def to_label_tensor(target):
    """
    Turns a luminessence label image into a 1D tensor, without normalizing it.
    Args:
        target (PIL.Image): input target
    Returns:
        image in Tensor type
    """
    return torch.from_numpy(np.array(target, dtype=np.uint8)).long()


def mask_to_class(mask, mapping):
    """Given the cityscapes dataset, this maps to a 0..classes numbers.
    This is because we are using a subset of all masks, so we have this "mapping" function.
    This mapping function is used to map all the standard ids into the smaller subset.

    Args:
        mask (torch.Tensor): the original label
        mapping (dict): new mapping of labels
    Returns:
        new label
    """
    maskimg = torch.zeros((mask.size()[0], mask.size()[1]), dtype=torch.uint8)
    for k in mapping:
        maskimg[mask == k] = mapping[k]
    return maskimg.long()


def mask_to_rgb(mask, mappingrgb):
    """
    Given the Cityscapes mask file, this converts the ids into rgb colors.
    This is needed as we are interested in a sub-set of labels, thus can't just use the
    standard color output provided by the dataset.
    """
    rgbimg = torch.zeros((3, mask.size()[0], mask.size()[1]), dtype=torch.uint8)
    for k in mappingrgb:
        rgbimg[0][mask == k] = mappingrgb[k][0]
        rgbimg[1][mask == k] = mappingrgb[k][1]
        rgbimg[2][mask == k] = mappingrgb[k][2]
    return rgbimg
