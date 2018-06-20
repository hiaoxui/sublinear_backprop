import torch

from test_.config import cfg

torch.manual_seed(9788)


lower = torch.nn.Embedding(
    embedding_dim=cfg.embedding_dim,
    num_embeddings=cfg.voc_size,
)
upper = torch.nn.Linear(
    in_features=(1+cfg.bi)*cfg.hidden_dim,
    out_features=cfg.voc_size,
)

from core.packer import Packer
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
    longest=cfg.max_step
)
if cfg.cuda:
    packer.cuda()

from test_.data_gen import Corpus
corpus = Corpus()


packer.train()
cnt = 0
optim = torch.optim.Adam(packer.parameters())

import time

xs, ys = corpus(cfg.batch_size)
xs, ys = torch.tensor(xs, dtype=torch.int64), torch.tensor(ys, dtype=torch.int64)
if cfg.cuda:
    xs, ys = xs.cuda(), ys.cuda()

for longest in range(12, -1, -1):
    longest = 2 ** longest
    print('longest = {}'.format(longest))
    packer.step = longest

    packer.eval()

    since = time.clock()
    for _0 in range(cfg.n_iter * 10):
        output = packer(xs)
        if cfg.print_out and _0 == 0:
            print((ys == output).sum().cpu().numpy())
    print('time overhead: {:.5f}'.format(time.clock() - since))

    # input('continue?')

    packer.train()

    since = time.clock()
    for _ in range(cfg.n_iter):
        optim.zero_grad()
        packer(xs, ys)
        optim.step()

    print('time overhead: {:.5f}'.format(time.clock() - since))

    # input('continue?')
