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
    longest=cfg.max_step
)
if cfg.cuda:
    packer.cuda()

from data_gen import Corpus
corpus = Corpus()


packer.train()
cnt = 0
optim = torch.optim.Adam(packer.parameters())
while True:
    xs, ys = corpus(cfg.batch_size)
    xs, ys = torch.tensor(xs), torch.tensor(ys)
    if cfg.cuda:
        xs, ys = xs.cuda(), ys.cuda()
    if cnt % 10 == 0:
        packer.eval()
        outputs = packer(xs)
        outputs = outputs.contiguous().view(-1)
        ys = ys.view(-1)
        print((outputs == ys).sum().cpu().numpy() / len(outputs))
        packer.train()
    else:
        optim.zero_grad()
        packer(xs, ys)
        optim.step()

    print('-'*25 + str(cnt) + '-'*25)
    cnt += 1
