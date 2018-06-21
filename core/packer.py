import torch

from .bitree import BiTree
from .mega_cell import MegaCell
from .utils import *


class Packer(torch.nn.Module):
    """
    Packer is a bad name, but my English is poor.
    Packer could serve as a substitute of torch.nn.Module, but it could only used for RNNs models.
    You cannot use Packer as a component of another torch.nn.Module, since it is in fact not a
    subclass of torch.nn.Module. You can only use it to do estimation or inference.
    For ultra long sequence, packer could make a trade-off between computational time and memory.
    Packer doesn't simply allocate all the memory needed for an RNNs sequence. Instead, we use a
    clever scheme for forward and backward propagation of RNNs. For more details:
    https://timvieira.github.io/blog/post/2016/10/01/reversing-a-sequence-with-sublinear-space/

    Usage example::

    >>> # binary classification, voc_size = 30, emb_dim = 10, hidden_dim = 20, batch first
    >>> packer = Packer(partial(torch.nn.LSTMCell, 10, 20), torch.nn.Linear(20, 2),
    >>>                 torch.nn.Embedding(30, 10), [20, 20], True, False,
    >>>                 torch.nn.CrossEntropyLoss(), 16)
    >>> xs = torch.randint(0, 30, size=(17, 4096), dtype=torch.int64) # batch_size = 17
    >>> ys = torch.randint(0, 2 , size=(17, 4096), dtype=torch.int64)  # binary classification
    >>> # estimation
    >>> packer.train()
    >>> from torch.optim import Adam
    >>> optimizer = Adam(packer.parameters())
    >>> optimizer.zero_grad()
    >>> packer.forward(xs, ys)
    >>> optimizer.step()
    >>> # inference
    >>> packer.eval()
    >>> output = packer.forward(xs, argmax=True).view(-1)
    >>> ys = ys.view(-1)
    >>> print('acc = {}'.format((ys==output).sum() / len(ys)))

    """
    def __init__(self, cell_constructor, upper, lower,
                 hidden_size, batch_first, bidirectional,
                 criterion, longest):
        """
        :param func cell_constructor: Constructor for RNNs cell. You should specify
        all the parameters of the cell outside. Usage Example:
        >>> partial(torch.nn.LSTMCell, 10, 20)
        >>> partial(torch.nn.GRUCell, 10, 20)
        :param torch.nn.Module upper: (optional) The outputs of RNNs cells will be sent
        to module upper. Note that you need to give us a module, instead of a constructor.
        Usage Example:
        >>> torch.nn.Linear(20, 2) # (For a binary classification)
        :param torch.nn.Module lower: (optional) All the inputs will be preprocessed by
        module lower before being sent to RNNs cells. Note that you need to give us a
        module, instead of a constructor.
        Usage Example:
        >>> torch.nn.Embedding(30, 10) # (for voc_size = 30 and emb_dim = 10)
        :param int/list[int] hidden_size: If the RNNs cell contains only one type of hidden states,
        set hidden_size as its dimensional. For example, for torch.nn.GRU(10, 20), you should set
        hidden_size=20. If the RNNs cell have multiple types of hidden states, like LSTMs, you
        should specify each dimension of them in a python list. For example, if the cell is
        torch.nn.LSTM(10, 20), you should set hidden_size=[20, 20].
        Note that we will always take the first hidden state in the list as the input to upper
        layer. For example, if you specify hidden_size=[11, 12, 13], the upper layer could be set
        as torch.nn.Linear(11, 2).
        :param bool batch_first: If True, the inputs (xs) should be shaped as (batch, seq, feature).
        :param bool bidirectional: If True, bidirectional.
        :param torch.nn.modules.loss._Loss criterion: We will not bother you to calculate the loss
        yourself, but you could tell us how to calculate the loss.
        Usage Example:
        >>> torch.nn.CrossEntropyLoss
        :param int longest: The inputs will chunk'ed into several parts, and ``longest`` is the
        maximum length of each chunk. The larger of ``longest``, the faster, but the more memory
        will be occupied.
        """
        super(Packer, self).__init__()

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

    def _init_state(self, batch_size, xs, hidden_size=None):
        """
        Initial state, namely, h0.
        :param int batch_size: Batch size.
        :param torch.Tensor xs: Input tensor.
        :param int or list[int] hidden_size: The dimension of hidden states.
        :rtype list[torch.Tensor]/torch.Tensor:
        """
        hidden_size = hidden_size or self.hidden_size

        if isinstance(hidden_size, int):
            return xs.new_zeros(size=(batch_size, hidden_size), dtype=torch.float32)
        else:
            rst = list()
            for hidden_size_ in hidden_size:
                rst.append(xs.new_zeros(size=(batch_size, hidden_size_), dtype=torch.float32))
            return rst

    def _inference(self, xs, argmax):
        """
        Inference.
        :param tuple[torch.Tensor] xs: Chunk'ed input.
        :param bool argmax: Whether to apply argmax on the last dimension of outputs.
        :rtype: torch.Tensor
        """
        all_output = list()
        batch_size = xs[0].shape[1]
        first_state = self._init_state(batch_size, xs[0])
        if self.bidirectional:
            n_chunk = len(xs)

            r2l_tree = BiTree(total=n_chunk,
                              first_state=first_state,
                              stepper=lambda hidden_state_, time_stamp_:
                              self.mega_cell.forward(xs[n_chunk-time_stamp_-1], -1,
                                                     h0_r2l=hidden_state_))
            r2l_tree.forward()

            l2r_state = first_state
            for chunk_idx, r2l_state in enumerate(r2l_tree.backward_generator()):
                l2r_state, _ = \
                    self.mega_cell.forward(xs[chunk_idx], 0, h0_l2r=l2r_state, h0_r2l=r2l_state)
                all_output.append(self.mega_cell.get_output(argmax))

        else:
            curr_hidden = first_state
            for xs_chunk in xs:
                curr_hidden = self.mega_cell.forward(
                    xs_chunk,
                    1, h0_l2r=curr_hidden)
                all_output.append(self.mega_cell.get_output(argmax))
        all_output = torch.cat(all_output, dim=0)
        return all_output.contiguous()

    def _estimate(self, xs, ys, seq_len):
        """
        :param tuple(torch.Tensor) xs: Features
        :param tuple(torch.Tensor) ys: Labels.
        :param int seq_len: Length of the sequence.
        """
        batch_size = xs[0].shape[1]
        first_state = self._init_state(batch_size, xs[0])
        n_chunk = len(xs)

        def extract_grad(state):
            """
            :param torch.Tensor or list[torch.Tensor] state:
            :rtype: torch.Tensor or list[torch.Tensor]
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

            def gen_r2l_stepper(h0_l2r_):
                def r2l_stepper(h0_r2l_, time_stamp_):
                    x = xs[n_chunk-time_stamp_-1]
                    state_l2r_, state_r2l_ = self.mega_cell.forward(x, 0, h0_l2r=h0_l2r_,
                                                                    h0_r2l=h0_r2l_)
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
                    r2l_gen.send(gen_r2l_stepper(left_state))
                except StopIteration:
                    pass
                time_stamp = n_chunk - 1 - chunk_idx
                ys_ = ys[time_stamp]
                self.mega_cell.backward(ys_, 1, right_grad, loss_weight=len(ys_)/seq_len)
                right_grad = extract_grad(left_state)

            # in case of repeated gradients computation of upper layers
            self.mega_cell.zero_upper_grad()

            left_state = first_state
            left_grad = None
            for chunk_idx, right_state in enumerate(r2l_tree.backward_generator()):
                if chunk_idx == 0:
                    continue
                time_stamp = chunk_idx - 1
                retain_grad(right_state)
                xs_, ys_ = xs[time_stamp], ys[time_stamp]
                left_state, _ = self.mega_cell.forward(xs_, 0, h0_l2r=left_state, h0_r2l=right_state)
                self.mega_cell.backward(ys_, -1, additional_grad=left_grad,
                                        loss_weight=len(xs_)/seq_len)
                left_grad = extract_grad(right_state)

        else:
            right_grad = None
            for chunk_idx, left_state in enumerate(l2r_tree.backward_generator()):
                time_stamp = n_chunk - chunk_idx - 1
                retain_grad(left_state)
                xs_, ys_ = xs[time_stamp], ys[time_stamp]
                self.mega_cell.forward(xs_, 1, h0_l2r=left_state)
                self.mega_cell.backward(ys_, 1, additional_grad=right_grad,
                                        loss_weight=len(xs_) / seq_len)
                right_grad = extract_grad(left_state)

    def forward(self, xs, ys=None, argmax=True):
        """
        :param torch.Tensor xs: Features.
        :param torch.Tensor ys: Labels.
        :param bool argmax: Whether to apply argmax on the last dimension of outputs. Only
        applicable on the inference mode.
        :rtype: None or torch.Tensor
        """
        if self.batch_first:
            xs = xs.transpose(1, 0)
            if ys is not None:
                ys = ys.transpose(1, 0)
        seq_len = len(xs)
        n_chunk = (len(xs)-1) // self.step + 1
        xs = xs.chunk(n_chunk, dim=0)
        if ys is not None:
            ys = ys.chunk(n_chunk, dim=0)
        if self.training and ys is None:
            raise Exception("Please provide labels during training!")
        if self.training:
            self._estimate(xs, ys, seq_len)
            self.mega_cell.reset()
        else:
            output = self._inference(xs, argmax)
            self.mega_cell.reset()
            if self.batch_first:
                return output.transpose(1, 0)
            else:
                return output
