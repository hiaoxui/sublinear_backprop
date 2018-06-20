from test_.built_in_lstm import OriginalLSTM
from test_.trainer import Trainer
import time
from test_.config import cfg
from test_.data_gen import Corpus
import torch


def main():
    model = OriginalLSTM()
    tr = Trainer(model)

    corpus = Corpus()
    xs, ys = corpus(cfg.batch_size)
    xs = torch.tensor(xs, dtype=torch.int64)
    ys = torch.tensor(ys, dtype=torch.int64)

    if cfg.cuda:
        xs = xs.cuda()
        ys = ys.cuda()

    since = time.time()
    tr.evaluate(cfg.n_iter * 10, xs, ys)
    print(time.time() - since)

    since = time.time()
    tr.train(cfg.n_iter, xs, ys)
    print(time.time() - since)

    tr.evaluate(cfg.n_iter, xs, ys)


if __name__ == '__main__':
    main()

