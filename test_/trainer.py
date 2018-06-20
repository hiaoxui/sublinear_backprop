from test_.data_gen import Corpus
from test_.config import cfg
import torch.nn as nn
from torch import optim
import torch
from tqdm import tqdm
import numpy as np
import time


class Trainer(object):
    def __init__(self, model):
        """

        :param nn.Module model:
        """
        self.model = model
        self.corpus = Corpus()
        self.optim = optim.Adam(self.model.parameters())
        if cfg.cuda:
            self.model.cuda()

    def train(self, n_iter, xs, ys):
        self.model.train()
        criterion = nn.CrossEntropyLoss()
        for _ in range(n_iter):
            output = self.model(xs).view(-1, cfg.voc_size)
            ys = ys.view(-1)
            loss = criterion(output, ys)
            self.model.zero_grad()
            loss.backward()
            self.optim.step()

    def evaluate(self, n_iter, xs, ys):
        self.model.eval()
        for _0 in range(n_iter):
            output = self.model(xs).view(-1, cfg.voc_size).data.cpu().numpy()
            if cfg.print_out and _0 == 0:
                print((np.argmax(output, axis=-1) == ys.data.cpu().numpy().reshape(-1)).sum())
