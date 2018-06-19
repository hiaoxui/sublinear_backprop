import torch


def detach_tensor(tensors):
    """
    Detach tensors.
    :param list or tuple or torch.Tensor tensors:
    :rtype: torch.Tensor or list[torch.Tensor]
    """
    if isinstance(tensors, list) or isinstance(tensors, tuple):
        return [detach_tensor(tensor) for tensor in tensors]
    return tensors.detach()


def retain_grad(tensors):
    """
    Retain gradients.
    :param list or tuple or torch.Tensor tensors:
    """
    if isinstance(tensors, list) or isinstance(tensors, tuple):
        for tensor in tensors:
            retain_grad(tensor)
        return
    tensors.requires_grad_()
    tensors.retain_grad()
