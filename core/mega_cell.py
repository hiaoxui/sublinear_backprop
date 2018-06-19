import torch

from .utils import *


class MegaCell(object):
    """
    MegaCell is an abstract component for recurrent neural networks (RNNs).
    It contains three torch Modules, a Module that preprocess inputs before sending
    to RNNs (optional), a RNN cell and a prediction layer (optional).
    Basically speaking, it acts like a typical RNNs prediction model, processing
    inputs with multiple timestamps bidirectionally, while it is not
    inherited from torch.Module.
    Besides the general functions of RNNs, MegaCell also allows users to specify
    initiate hidden states during the forward propagation, and provide the gradients
    from the previous MegaCell during the backward propagation, which is essential
    for sublinear backpropagation.
    Due to some theoretic problems, MegaCell doesn't support multilayer RNNs.
    """
    def __init__(self, base_cell_constructor, upper, lower,
                 bidirectional, criterion, multi_hidden):
        """
        :param base_cell_constructor: Constructor for RNNs cell. You should specify
        all the parameters of the cell outside. Usage Examples:
        >>> partial(torch.nn.LSTMCell, 10, 20)
        >>> partial(torch.nn.GRUCell, 10, 20)
        :param bool bidirectional: If True, two cells will be initiated.
        :param torch.nn.Module upper: (optional) The outputs of RNNs cells will be sent
        to module upper. Note that you need to give us a module, instead of a constructor.
        Usage Example:
        >>> torch.nn.Linear(20, 2) # (For a binary classification)
        :param torch.nn.Module lower: (optional) All the inputs will be preprocessed by
        module lower before being sent to RNNs cells. Note that you need to give us a
        module, instead of a constructor.
        Usage Example:
        >>> torch.nn.Embedding(30, 10) # (for voc_size = 30 and emb_dim = 10)
        :param torch.nn.modules.loss._Loss criterion: How to calculate the loss?
        Usage Example:
        >>> torch.nn.CrossEntropyLoss()
        :param bool multi_hidden: Does the cell have multiple types of hidden states?
        Example: For LSTMs, it's True because LSTMs have both hidden states and memory cell.
        Example: For GRUs, it's False since it contains only hidden states.
        """
        self.upper = upper
        self.lower = lower

        self.bidirectional = bidirectional
        self.multi_hidden = multi_hidden

        self.cell_l2r = base_cell_constructor()
        if self.bidirectional:
            self.cell_r2l = base_cell_constructor()
        else:
            self.cell_r2l = None

        self.states_l2r = list()
        self.states_r2l = list()

        self.criterion = criterion

    def reset(self):
        """
        Reset all calculated states.
        You don't need to call this method very often since we will clear it when they are no
        longer needed.
        """
        del self.states_l2r[:]
        del self.states_r2l[:]

    def forward(self, time_steps, direction, *, h0_l2r=None, h0_r2l=None):
        """
        Forward propagation. All the intermediate hidden states will be stored in the class.
        :param torch.Tensor time_steps: Inputs, or xs. Must be the shape of (sequence, batch, ...)
        :param int direction: -1 (Right to left), 0 (Both directions) or 1 (Left to right).
        :param torch.Tensor h0_l2r: Hidden states for left2right propagation.
        Needed if direction = 1 or 0.
        :param torch.Tensor h0_r2l: Hidden states for right2right propagation.
        Needed if direction = -1 or 0.
        :return: Last hidden state of forward propagation.
        """
        def _forward_helper(cell, time_steps_, h0):
            """
            :param torch.nn.Module cell: Cell to use.
            :param torch.Tensor time_steps_: Inputs.
            :param torch.Tensor h0: Initial hidden states.
            :return: All the intermediate hidden states.
            """
            curr_state = h0
            states = list()
            for time_stamp in time_steps_:
                curr_state = cell(time_stamp, curr_state)
                states.append(curr_state)
            return states

        assert h0_r2l is None or self.bidirectional
        assert direction in [-1, 0, 1]
        if self.lower is not None:
            time_steps = self.lower(time_steps)
        if direction >= 0:
            self.states_l2r = _forward_helper(self.cell_l2r, time_steps, h0_l2r)
        if direction <= 0:
            self.states_r2l = _forward_helper(self.cell_r2l, time_steps, h0_r2l)[::-1]
        if direction == -1:
            return detach_tensor(self.states_r2l[0])
        elif direction == 0:
            return detach_tensor([self.states_l2r[-1], self.states_r2l[0]])
        else:
            return detach_tensor(self.states_l2r[-1])

    def get_output(self, argmax):
        """
        Get all the outputs. Called after calling forward.
        :param bool argmax: Whether to use argmax at the last dimension.
        If True, more memory could be saved.
        :rtype: torch.Tensor
        """
        assert len(self.states_l2r) > 0
        assert not self.bidirectional or len(self.states_r2l) > 0
        states = list()
        if self.bidirectional:
            if self.multi_hidden:
                for state_l2r, state_r2l in zip(self.states_l2r, self.states_r2l):
                    states.append(torch.cat([state_l2r[0], state_r2l[0]], dim=-1))
            else:
                for state_l2r, state_r2l in zip(self.states_l2r, self.states_r2l):
                    states.append(torch.cat([state_l2r, state_r2l], dim=-1))
        else:
            if self.multi_hidden:
                for state_l2r in self.states_l2r:
                    states.append(state_l2r[0])
            else:
                states = self.states_l2r
        states = [state_t.unsqueeze(0) for state_t in states]
        states = torch.cat(states, dim=0)
        logits = self.upper(states)
        if argmax:
            return torch.argmax(logits, dim=-1)
        else:
            return logits

    def backward(self, ys, direction, additional_grad=None, loss_weight=1.0):
        """
        Backward propagation. Should be called after calling forward method.
        :param torch.Tensor ys: Labels. Must be shaped as (sequence, batch)
        :param int direction: -1 or 1. -1 for right2left, and 1 for left2right.
        :param torch.Tensor/list[torch.Tensor] additional_grad:
        Gradients from following layers. Should be shaped the same as hidden states.
        Leave it None if you have no additional gradients. E.g., for the last time stamp, there
        should be no gradients propagated from following cells.
        :param float loss_weight: Loss function often applies average over training examples. When
        the training examples are not the complete data, we need to add a factor to re-balance the
        loss.
        """

        def backward_helper(last_state_):
            """
            :param torch.Tensor or list last_state_: The state of the last time stamp.
            """
            losses = list()
            if additional_grad is not None:
                if self.multi_hidden:
                    for hidden_tensor, grad_ in zip(last_state_, additional_grad):
                        # It is equivalent mathematically.
                        losses.append((hidden_tensor * grad_).sum())
                else:
                    losses.append((last_state_ * additional_grad).sum())
            pred = self.get_output(argmax=False)
            ys_flat = ys.contiguous().view(-1)
            pred_flat = pred.view(len(ys_flat), -1)
            losses.append(self.criterion(pred_flat, ys_flat) * loss_weight)
            torch.autograd.backward(losses)

        assert direction in [-1, 1]

        if direction == 1:
            last_state = self.states_l2r[-1]
            backward_helper(last_state)
        else:
            last_state = self.states_r2l[0]
            backward_helper(last_state)

    def zero_upper_grad(self):
        """
        Zero the gradients of upper module.
        """
        self.upper.zero_grad()
