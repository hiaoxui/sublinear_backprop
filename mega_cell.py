import torch


class MegaCell(object):
    def __init__(self, base_cell_constructor, upper, lower,
                 bidirectional, criterion, multi_hidden):
        """

        :param base_cell_constructor:
        :param bool bidirectional:
        :param torch.nn.Module upper:
        :param torch.nn.Module lower:
        :param torch.nn.modules.loss._Loss criterion:
        :param bool multi_hidden:
        """
        self.upper = upper
        self.lower = lower

        self.bidirectional = bidirectional
        self.multi_hidden = multi_hidden

        assert issubclass(base_cell_constructor, torch.nn.Module)
        self.cell_l2r = base_cell_constructor()
        if self.bidirectional:
            self.cell_r2l = base_cell_constructor()

        self.states_l2r = list()
        self.states_r2l = list()

        self.criterion = criterion

    def reset(self):
        """

        :rtype: None
        """
        self.states_r2l = list()
        self.states_l2r = list()

    def forward(self, time_steps, direction, h0_l2r=None, h0_r2l=None):
        """

        :param torch.Tensor time_steps:
        :param int direction: -1, 0 or 1
        :param torch.Tensor h0_l2r:
        :param torch.Tensor h0_r2l:
        :return:
        """
        def _forward_helper(cell, time_steps_, h0):
            """

            :param torch.nn.Module cell:
            :param torch.Tensor time_steps_:
            :param torch.Tensor h0:
            :return:
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
            return self.states_r2l[0]
        elif direction == 0:
            return self.states_l2r[-1], self.states_r2l[0]
        else:
            return self.states_l2r[0]

    def get_output(self):
        """

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
                    states.append(state_l2r)
            else:
                states = self.states_l2r
        states = torch.cat(states)
        logits = self.upper(states)
        return logits

    def backward(self, direction, ys, additional_grad=None):
        """

        :param int direction: -1, 0 or 1
        :param torch.Tensor ys:
        :param torch.Tensor/list[torch.Tensor] additional_grad:
        :return:
        """

        def detach_states(states_):
            """

            :param list[torch.Tensor]/list[list[torch.Tensor]] states_:
            :rtype: None
            """
            if self.multi_hidden:
                for state_ in states_:
                    for hidden_state_ in state_:
                        hidden_state_.detach_()
            else:
                for state_ in states_:
                    state_.detach_()

        def need_grad(state_):
            """

            :param torch.Tensor/list[torch.Tensor] state_:
            :rtype:
            """
            if self.multi_hidden:
                for hidden_state_ in state_:
                    hidden_state_.requires_grad_()
                    hidden_state_.retain_grad()
            else:
                state_.requires_grad_()
                state_.retain_grad()

        def extract_grad(state):
            """

            :param torch.Tensor/list[torch.Tensor] state:
            :rtype:
            """
            if self.multi_hidden:
                return [hidden_state_.grad.detach() for hidden_state_ in state]
            else:
                return state.grad.detach()

        def backward_helper(last_state_):
            """

            :param torch.Tensor last_state_:
            :param torch.list[torch.Tensor] states_:
            :rtype: None
            """
            losses = list()
            if additional_grad is not None:
                if self.multi_hidden:
                    for hidden_tensor, grad_ in zip(last_state_, additional_grad):
                        losses.append((hidden_tensor * grad_).sum())
                else:
                    losses.append((last_state_ * additional_grad).sum())
            pred = self.get_output()
            losses.append(self.criterion(pred, ys))
            torch.autograd.backward(losses)

        assert direction in [-1, 1]
        if direction == 1:
            detach_states(self.states_r2l)
            need_grad(self.states_l2r[0])
            last_state = self.states_l2r[-1]
            backward_helper(last_state)
            return extract_grad(self.states_l2r[0])
        else:
            detach_states(self.states_l2r)
            need_grad(self.states_l2r[-1])
            last_state = self.states_r2l[0]
            backward_helper(last_state)
            return extract_grad(self.states_r2l[-1])

    def zero_upper_grad(self):
        """

        :rtype: None
        """
        self.upper.zero_grad()
