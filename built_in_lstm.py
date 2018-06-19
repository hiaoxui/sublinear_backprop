# from torch.nn import LSTM
from SemiLSTM import SemiLSTM as LSTM
from torch import nn
from torch.autograd import Variable
from config import cfg


class OriginalLSTM(nn.Module):
    def __init__(self):
        super(OriginalLSTM, self).__init__()
        self.encoder = nn.Embedding(
            embedding_dim=cfg.embedding_dim,
            num_embeddings=cfg.voc_size,
        )

        self.rnn = LSTM(
            input_size=cfg.embedding_dim,
            hidden_size=cfg.hidden_dim,
            batch_first=True,
            # bidirectional=cfg.bi,
        )

        self.fc = nn.Linear(
            in_features=cfg.hidden_dim * (1+cfg.bi),
            out_features=cfg.voc_size,
        )

    def forward(self, x):
        """
        :param Variable x: [batch_size, seq_length]
        :return:
        """
        code = self.encoder(x)
        lstm_out = self.rnn(code)
        lstm_out = lstm_out[0]
        output = self.fc(lstm_out)
        return output


if __name__ == '__main__':
    model = OriginalLSTM
