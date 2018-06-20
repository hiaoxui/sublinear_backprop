import torch


class HandMadeLSTM(torch.nn.Module):
    def __init__(self, input_size, hidden_size, batch_first, bidirectional=False):
        super(HandMadeLSTM, self).__init__()
        self.batch_first = batch_first
        self.cell_l2r = torch.nn.LSTMCell(input_size=input_size, hidden_size=hidden_size)
        if bidirectional:
            self.cell_r2l = torch.nn.LSTMCell(input_size=input_size, hidden_size=hidden_size)
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional

    def forward(self, xs):
        """

        :param torch.Tensor xs:
        :return:
        """
        if self.batch_first:
            xs = xs.transpose(1, 0)
        h = c = xs.new_zeros(size=(len(xs[1]), self.hidden_size,))
        rst = list()
        for x in xs:
            h, c = self.cell_l2r(x, (h, c))
            rst.append(h)
        if self.bidirectional:
            h = c = xs.new_zeros(size=(len(xs[1]), self.hidden_size,))
            length = len(xs)
            for l_idx in range(length-1, -1, -1):
                h, c = self.cell_r2l(xs[l_idx], (h, c))
                rst[l_idx] = torch.cat([rst[l_idx], h], dim=1)
        rst = [item.unsqueeze(dim=0) for item in rst]
        rst = torch.cat(rst, dim=0)
        if self.batch_first:
            rst = rst.transpose(1, 0)
        return rst, (None, None)
