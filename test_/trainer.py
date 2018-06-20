from test.data_gen import Corpus
from test.config import cfg
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

    def train(self, n_iter):
        self.model.train()
        criterion = nn.CrossEntropyLoss()
        bar = tqdm(total=n_iter)
        for _ in range(n_iter):
            xs, ys = self.corpus(cfg.batch_size)
            xs = torch.tensor(xs, dtype=torch.int64)
            ys = torch.tensor(ys, dtype=torch.int64)
            if cfg.cuda:
                xs = xs.cuda()
                ys = ys.cuda()
            output = self.model(xs).view(-1, cfg.voc_size)
            ys = ys.view(-1)
            loss = criterion(output, ys)
            self.model.zero_grad()
            loss.backward()
            self.optim.step()
            bar.update()
        bar.close()
        time.sleep(0.1)

    def evaluate(self, n_iter):
        self.model.eval()
        bar = tqdm(total=n_iter)
        t = f = 0
        for _ in range(n_iter):
            xs, ys = self.corpus(cfg.batch_size)
            xs = torch.tensor(xs, dtype=torch.int64)
            if cfg.cuda:
                xs = xs.cuda()
            output = self.model(xs).view(-1, cfg.voc_size).data.cpu().numpy()
            pred = np.argmax(output, axis=1)
            ys = ys.reshape(-1)
            t += (pred == ys).sum()
            f += (pred != ys).sum()
            bar.update()
        bar.close()

        time.sleep(0.1)
        print('T   : %d' % t)
        print('F   : %d' % f)
        print('Acc : %.4f' % (t/(t+f)))
        time.sleep(0.1)
