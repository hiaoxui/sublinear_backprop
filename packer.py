import torch

from mega_cell import MegaCell


class Packer(object):
    def __init__(self, cell_constructor, upper, lower, hidden_size, batch_first, bidirectional):
        self.mega_cell = MegaCell(
            base_cell=cell_constructor,
            upper=upper,
            lower=lower,
            batch_first=batch_first,
            multi_hidden=True
        )
