import torch


class Config(object):
    def __init__(self):
        self.hidden_dim = 20
        self.bi = True
        self.batch_size = 10
        self.repeat_period = 10
        self.voc_size = 25
        self.embedding_dim = 21
        self.max_step = 100

        self.seq_length = 2 ** 12

        self.n_iter = 1

        self.print_out = True

        self.cuda = False
        if True and torch.cuda.device_count() > 0:
            print('cuda detected!')
            self.cuda = True


cfg = Config()
