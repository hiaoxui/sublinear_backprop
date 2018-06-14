import torch

from mega_cell import MegaCell


class Packer(object):
    def __init__(self, cell_constructor, upper, lower,
                 hidden_size, batch_first, bidirectional,
                 criterion, longest):
        """

        :param cell_constructor:
        :param torch.nn.Module upper:
        :param torch.nn.Module lower:
        :param int/list[int] hidden_size:
        :param bool batch_first:
        :param bool bidirectional:
        :param torch.nn.modules.loss._Loss criterion:
        :param int longest:
        """
        self.multi_hidden = not isinstance(hidden_size, int)
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.batch_first = batch_first
        self.step = longest

        self.mega_cell = MegaCell(
            base_cell_constructor=cell_constructor,
            upper=upper,
            lower=lower,
            multi_hidden=self.multi_hidden,
            criterion=criterion,
            bidirectional=bidirectional
        )

        if self.multi_hidden:
            self.init_hidden = list()
            for n_hidden in hidden_size:
                self.init_hidden.append(torch.Tensor.zeros(size=(n_hidden,)))
        else:
            self.init_hidden = torch.zeros(size=(hidden_size,))

        self.training = False

    def eval(self):
        self.training = False

    def train(self):
        self.training = True

    def __call__(self, *args):
        self.forward(*args)

    def inference(self, xs):
        length = len(xs)
        all_output = list()
        if not self.bidirectional:
            raise NotImplementedError
        else:
            curr_idx = 0
            curr_hidden = self.init_hidden
            while curr_idx < length:
                curr_hidden = self.mega_cell.forward(
                    xs[curr_idx:curr_idx+self.step],
                    1, h0_l2r=curr_hidden)
                all_output.append(self.mega_cell.get_output())
        all_output = torch.cat(all_output, dim=0)
        return all_output

    def estimate(self, xs, ys):
        pass

    def forward(self, xs, ys=None):
        """

        :param torch.Tensor xs:
        :param torch.Tensor ys:
        :return:
        """
        if self.batch_first:
            xs = xs.transpose(0, 1)
            ys = ys.transpose(0, 1)
        if self.training and ys is None:
            raise Exception("Please provide labels during training!")
        if self.training:
            self.estimate(xs, ys)
        else:
            output = self.inference(xs)
            if self.batch_first:
                return output.transpose(0, 1)
            else:
                return output
