import torch


class Config(object):
    def __init__(self):
        self.hidden_dim = 64
        self.bi = False
        self.batch_size = 32
        self.repeat_period = 16
        self.voc_size = 64
        self.embedding_dim = 16

        self.seq_length = 256

        self.cuda = False
        if torch.cuda.device_count() > 0:
            print('cuda detected!')
            self.cuda = True


cfg = Config()
