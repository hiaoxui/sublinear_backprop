import torch

from config import cfg


lower = torch.nn.Embedding(
    embedding_dim=cfg.embedding_dim,
    num_embeddings=cfg.voc_size,
)
upper = torch.nn.Linear(
    in_features=(1+cfg.bi)*cfg.hidden_dim,
    out_features=cfg.voc_size,
)

from packer import Packer
from functools import *

packer = Packer(
    cell_constructor=partial(torch.nn.LSTMCell,
    #cell_constructor=partial(torch.nn.GRUCell,
                             input_size=cfg.embedding_dim,
                             hidden_size=cfg.hidden_dim
                             ),
    upper=upper,
    lower=lower,
    batch_first=True,
    hidden_size=[cfg.hidden_dim, cfg.hidden_dim],
    #hidden_size=cfg.hidden_dim,
    bidirectional=cfg.bi,
    criterion=torch.nn.CrossEntropyLoss(),
    longest=200
)

from data_gen import Corpus
corpus = Corpus()


packer.eval()
cnt = 0
while True:
    xs, ys = corpus(cfg.batch_size)
    xs, ys = torch.tensor(xs), torch.tensor(ys)
    packer(xs)
    print('-' * 25 + str(cnt) + '-'*25)
    cnt += 1
