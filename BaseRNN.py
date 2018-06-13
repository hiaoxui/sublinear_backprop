import torch


class MegaCell(object):
    def __init__(self, base_cell, upper, lower, bidirectional, hidden_size, batch_first):
        """

        :param torch.nn.Module base_cell:
        :param bool bidirectional:
        :param torch.nn.Module upper:
        :param torch.nn.Module lower:
        :param list[int] hidden_size:
        """
        self.base_cell = base_cell
        self.upper = upper
        self.lower = lower

        if isinstance(hidden_size, int):
            hidden_size = [hidden_size]
        self.hidden_size = hidden_size

        self.batch_first = batch_first
        self.bidirectional = bidirectional

    def _forward(self):
        pass
