import torch


class SemiLSTM(torch.nn.Module):
    def __init__(self, input_size, hidden_size, batch_first):
        super(SemiLSTM, self).__init__()
        self.batch_first = batch_first
        self.cell = torch.nn.LSTMCell(input_size=input_size, hidden_size=hidden_size)
        self.hidden_size = hidden_size

    def forward(self, xs):
        """

        :param torch.Tensor xs:
        :return:
        """
        if self.batch_first:
            xs = xs.transpose(1, 0)
        h = c = torch.zeros(size=(len(xs[1]), self.hidden_size,), device=torch.device(0))
        rst = list()
        for x in xs:
            h, c = self.cell(x, (h, c))
            rst.append(h)
        rst = [item.unsqueeze(dim=0) for item in rst]
        rst = torch.cat(rst, dim=0)
        if self.batch_first:
            rst = rst.transpose(1, 0)
        return rst, (h, c)
