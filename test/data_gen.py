from test.config import cfg
import numpy as np


class Corpus(object):
    def __init__(self, seed=9788):
        self.seed = seed

    def __call__(self, n_seq):
        np.random.seed(self.seed)
        self.seed += 1
        xs = np.random.randint(1, cfg.voc_size, size=(n_seq, cfg.seq_length))
        ys = np.zeros(shape=(n_seq, cfg.seq_length), dtype=np.int64)
        ys[:, cfg.repeat_period:] = xs[:, :-cfg.repeat_period]
        return xs, ys
