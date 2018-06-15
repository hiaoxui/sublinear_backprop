import torch


def detach_tensor(tensors):
    """

    :param torch.Tensor/list[torch.Tensor] tensors:
    :rtype: torch.Tensor/list[torch.Tensor]
    """
    if isinstance(tensors, list) or isinstance(tensors, tuple):
        return [detach_tensor(tensor) for tensor in tensors]
    return tensors.detach()


class Scope(object):
    def __init__(self, beg, end):
        self.beg = beg
        self.end = end

    def __contains__(self, item):
        assert isinstance(item, int)
        return self.beg <= item < self.end

    def __getitem__(self, item_idx):
        if item_idx not in [0, 1]:
            raise IndexError
        return self.beg if item_idx == 0 else self.end
