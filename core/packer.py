import torch

from core.bitree import BiTree
from core.mega_cell import MegaCell
from utils import *


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

        self.training = False
        self.is_cuda = False

    def _init_state(self, batch_size, hidden_size=None):
        """

        :param int batch_size:
        :rtype list[torch.Tensor]/torch.Tensor:
        """
        hidden_size = hidden_size or self.hidden_size
        if isinstance(hidden_size, int):
            rst = torch.zeros(size=(batch_size, hidden_size))
            if self.is_cuda:
                return rst.cuda()
            else:
                return rst
        else:
            rst = list()
            for hidden_size_ in hidden_size:
                to_app = torch.zeros(batch_size, hidden_size_)
                if self.is_cuda:
                    rst.append(to_app.cuda())
                else:
                    rst.append(to_app)
            return rst

    def eval(self):
        self.training = False

        def helper(module):
            if module is not None and hasattr(module, 'eval'):
                module.eval()

        helper(self.mega_cell.upper)
        helper(self.mega_cell.lower)
        helper(self.mega_cell.cell_l2r)
        helper(self.mega_cell.cell_r2l)

    def train(self):
        self.training = True

        def helper(module):
            if module is not None and hasattr(module, 'train'):
                module.train()

        helper(self.mega_cell.upper)
        helper(self.mega_cell.lower)
        helper(self.mega_cell.cell_l2r)
        helper(self.mega_cell.cell_r2l)

    def __call__(self, *args):
        return self.forward(*args)

    def _inference(self, xs, argmax):
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

            r2l_tree = BiTree(total=n_chunk,
                              first_state=first_state,
                              stepper=lambda hidden_state, time_stamp:
                              self.mega_cell.forward(xs[n_chunk-time_stamp-1], -1, h0_r2l=hidden_state))
            r2l_tree.forward()

            l2r_state = first_state
            for chunk_idx, r2l_state in enumerate(r2l_tree.backward_generator()):
                l2r_state, _ = \
                    self.mega_cell.forward(xs[chunk_idx], 0, h0_l2r=l2r_state, h0_r2l=r2l_state)
                all_output.append(self.mega_cell.get_output(argmax))
                self.mega_cell.reset()

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

    def _estimate(self, xs, ys):
        batch_size = xs[0].shape[1]
        first_state = self._init_state(batch_size)
        n_chunk = len(xs)

        def extract_grad(state):
            """

            :param torch.Tensor or list[torch.Tensor] state:
            :rtype:
            """
            if self.multi_hidden:
                return [hidden_state_.grad.detach() for hidden_state_ in state]
            else:
                return state.grad.detach()

        l2r_tree = BiTree(total=n_chunk,
                          first_state=first_state,
                          stepper=lambda hidden_state_, time_stamp_:
                          self.mega_cell.forward(xs[time_stamp_], 1, h0_l2r=hidden_state_))
        l2r_tree.forward()

        if self.bidirectional:

            def r2l_stepper_gen(h0_l2r_):
                def r2l_stepper(h0_r2l_, time_stamp_):
                    x = xs[n_chunk-time_stamp_-1]
                    state_l2r_, state_r2l_ = self.mega_cell.forward(x, 0, h0_l2r=h0_l2r_, h0_r2l=h0_r2l_)
                    return state_r2l_
                return r2l_stepper

            r2l_tree = BiTree(total=n_chunk+1,
                              first_state=first_state,
                              stepper=lambda h0_r2l_, time_stamp_:
                              self.mega_cell.forward(xs[n_chunk-1-time_stamp_], -1, h0_r2l=h0_r2l_)
                              )

            r2l_gen = r2l_tree.forward_generator()
            next(r2l_gen)
            right_grad = None
            for chunk_idx, left_state in enumerate(l2r_tree.backward_generator()):
                retain_grad(left_state)
                try:
                    r2l_gen.send(r2l_stepper_gen(left_state))
                except StopIteration:
                    pass
                time_stamp = n_chunk - 1 - chunk_idx
                self.mega_cell.backward(ys[time_stamp], 1, right_grad)
                right_grad = extract_grad(left_state)
                self.mega_cell.reset()

            # in case of repeated gradients computation of upper layers
            self.mega_cell.zero_upper_grad()

            left_state = self._init_state(batch_size)
            left_grad = None
            for chunk_idx, right_state in enumerate(r2l_tree.backward_generator()):
                if chunk_idx == 0:
                    continue
                time_stamp = chunk_idx - 1
                retain_grad(right_state)
                self.mega_cell.forward(xs[time_stamp], 0, h0_l2r=left_state, h0_r2l=right_state)
                self.mega_cell.backward(ys[time_stamp], -1, additional_grad=left_grad)
                left_grad = extract_grad(right_state)
                self.mega_cell.reset()

        else:
            right_grad = None
            for chunk_idx, left_state in enumerate(l2r_tree.backward_generator()):
                time_stamp = n_chunk - chunk_idx - 1
                retain_grad(left_state)
                self.mega_cell.forward(xs[time_stamp], 1, h0_l2r=left_state)
                self.mega_cell.backward(ys[time_stamp], 1, additional_grad=right_grad)
                right_grad = extract_grad(left_state)

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
            self._estimate(xs, ys)
            self.mega_cell.reset()
        else:
            output = self._inference(xs, argmax)
            self.mega_cell.reset()
            if self.batch_first:
                return output.transpose(1, 0)
            else:
                return output

    def parameters(self):
        if self.mega_cell.lower is not None:
            yield from self.mega_cell.lower.parameters()
        if self.mega_cell.upper is not None:
            yield from self.mega_cell.upper.parameters()
        if self.mega_cell.cell_l2r is not None:
            yield from self.mega_cell.cell_l2r.parameters()
        if self.mega_cell.cell_r2l is not None:
            yield from self.mega_cell.cell_r2l.parameters()

    def cuda(self):
        if self.mega_cell.lower is not None:
            self.mega_cell.lower.cuda()
        if self.mega_cell.upper is not None:
            self.mega_cell.upper.cuda()
        if self.mega_cell.cell_l2r is not None:
            self.mega_cell.cell_l2r.cuda()
        if self.mega_cell.cell_r2l is not None:
            self.mega_cell.cell_r2l.cuda()
        self.is_cuda = True
