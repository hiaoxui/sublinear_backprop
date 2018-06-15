import torch
from bitree import BiTree

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
                self.init_hidden.append(torch.zeros(size=(n_hidden,)))
        else:
            self.init_hidden = torch.zeros(size=(hidden_size,))

        self.training = False

    def _init_state(self, batch_size, hidden_size=None):
        """

        :param int batch_size:
        :rtype list[torch.Tensor]/torch.Tensor:
        """
        hidden_size = hidden_size or self.hidden_size
        if isinstance(hidden_size, int):
            return self.init_hidden.unsqueeze(0).expand(batch_size, *self.init_hidden.shape)
        else:
            rst = list()
            for hidden_state in self.init_hidden:
                rst.append(hidden_state.unsqueeze(0).expand(batch_size, *hidden_state.shape))
            return rst

    def eval(self):
        self.training = False

    def train(self):
        self.training = True

    def __call__(self, *args):
        self.forward(*args)

    def inference(self, xs, argmax):
        """

        :param tuple[torch.Tensor] xs:
        :param bool argmax:
        :return:
        """
        all_output = list()
        batch_size = xs[0].shape[1]
        first_state = self._init_state(batch_size)
        if self.bidirectional:
            n_chunk = len(xs)

            r2l_tree = BiTree(total=n_chunk+1,
                              first_state=first_state,
                              stepper=lambda hidden_state, time_stamp:
                              self.mega_cell.forward(xs[n_chunk-time_stamp-1], -1, h0_r2l=hidden_state))
            r2l_tree.forward()

            reverse_gen = r2l_tree.backward_generator()
            next(reverse_gen)

            for chunk_idx in range(n_chunk):
                pass

        else:
            curr_hidden = first_state
            for xs_chunk in xs:
                curr_hidden = self.mega_cell.forward(
                    xs_chunk,
                    1, h0_l2r=curr_hidden)
                all_output.append(self.mega_cell.get_output(argmax))
                self.mega_cell.reset()
        all_output = torch.cat(all_output, dim=0)
        return all_output

    def estimate(self, xs, ys):
        pass

    def forward(self, xs, ys=None, argmax=True):
        """

        :param torch.Tensor xs:
        :param torch.Tensor ys:
        :param bool argmax:
        :return:
        """
        if self.batch_first:
            xs = xs.transpose(1, 0)
            if ys is not None:
                ys = ys.transpose(1, 0)
        n_chunk = (len(xs)-1) // self.step + 1
        xs = xs.chunk(n_chunk, dim=0)
        if ys is not None:
            ys = ys.chunk(n_chunk, dim=0)
        if self.training and ys is None:
            raise Exception("Please provide labels during training!")
        if self.training:
            self.estimate(xs, ys)
        else:
            output = self.inference(xs, argmax)
            if self.batch_first:
                return output.transpose(1, 0)
            else:
                return output
