import torch


class MegaCell(object):
    def __init__(self, base_cell_constructor, upper, lower,
                 bidirectional, hidden_size, batch_first,
                 criterion, multi_hidden):
        """

        :param base_cell_constructor:
        :param bool bidirectional:
        :param torch.nn.Module upper:
        :param torch.nn.Module lower:
        :param list[int] hidden_size:
        :param torch.nn.modules.loss._Loss criterion:
        :param bool multi_hidden:
        """
        self.upper = upper
        self.lower = lower

        if isinstance(hidden_size, int):
            hidden_size = [hidden_size]
        self.hidden_size = hidden_size

        self.batch_first = batch_first
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
        self.cell_l2r.zero_grad()
        self.cell_r2l.zero_grad()

    @staticmethod
    def _forward_helper(cell, time_steps, h0):
        """

        :param torch.nn.Module cell:
        :param torch.Tensor time_steps:
        :param torch.Tensor h0:
        :return:
        """
        curr_state = h0
        states = list()
        for time_stamp in time_steps:
            curr_state = cell(time_stamp, curr_state)
            states.append(curr_state)
        return states

    def forward(self, time_steps, direction, h0_l2r=None, h0_r2l=None):
        """

        :param torch.Tensor time_steps:
        :param str direction: l2r, r2l or bi
        :param torch.Tensor h0_l2r:
        :param torch.Tensor h0_r2l:
        :return:
        """
        assert bool(h0_r2l is None) or self.bidirectional
        if self.lower is not None:
            time_steps = self.lower(time_steps)
        if self.batch_first:
            time_steps = time_steps.transpose(1, 0)
        if direction == 'l2r' or direction == 'bi':
            self.states_l2r = self._forward_helper(self.cell_l2r, time_steps, h0_l2r)
        if direction == 'r2l' or direction == 'bi':
            self.states_r2l = self._forward_helper(self.cell_r2l, time_steps, h0_r2l)[::-1]

    def _backward_helper(self, additional_grad, last_state, states, ys):
        """

        :param torch.Tensor/list[torch.Tensor] additional_grad:
        :param torch.Tensor last_state:
        :param torch.list[torch.Tensor] states:
        :param torch.Tensor ys:
        :return:
        """
        losses = list()
        losses.append(self.upper(states))
        if isinstance(last_state, list):
            for state_, grad_ in zip(states, additional_grad):
                losses.append((state_ * grad_).sum())
        else:
            losses.append((last_state * additional_grad).sum())
        logits = list()
        for state_, y in zip(states, ys):
            logits.append(self.upper(state_))
        pred = torch.tensor(logits)
        losses.append(self.criterion(pred, ys))
        torch.autograd.backward(losses)

    def backward(self, direction, ys, grad_h0=None, grad_hn=None):
        """

        :param str direction: l2r, r2l or bi
        :param torch.Tensor ys:
        :param torch.Tensor/list[torch.Tensor] grad_h0:
        :param torch.Tensor/list[torch.Tensor] grad_hn:
        :return:
        """
        states = list()
        if self.bidirectional:
            if self.multi_hidden:
                for state_l2r, state_r2l in zip(self.states_l2r, self.states_r2l):
                    states.append(torch.cat([state_l2r[0], state_r2l[0]]))
            else:
                for state_l2r, state_r2l in zip(self.states_l2r, self.states_r2l):
                    states.append(torch.cat([state_l2r, state_r2l]))
        else:
            if self.multi_hidden:
                for state_l2r in self.states_l2r:
                    states.append(state_l2r)
            else:
                states = self.states_l2r

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

        def extract_grad(state_):
            """

            :param torch.Tensor/list[torch.Tensor] state_:
            :rtype:
            """
            if self.multi_hidden:
                return [hidden_state_.grad.detach() for hidden_state_ in state_]
            else:
                return state_.grad.detach()

        if direction == 'l2r':
            detach_states(self.states_r2l)
            need_grad(self.states_l2r[0])
            last_state = self.states_l2r[-1]
            self._backward_helper(grad_hn, last_state, states, ys)
            return extract_grad(self.states_l2r[0])
        elif direction == 'r2l':
            detach_states(self.states_l2r)
            need_grad(self.states_l2r[-1])
            last_state = self.states_r2l[0]
            self._backward_helper(grad_hn, last_state, states, ys)
            return extract_grad(self.states_r2l[-1])
        else:
            raise NotImplementedError

    def zero_upper_grad(self):
        """

        :rtype: None
        """
        self.upper.zero_grad()

    def first_state_l2r(self):
        pass

    def first_state_r2l(self):
        pass
