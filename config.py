import torch


class Config(object):
    def __init__(self):
        self.hidden_dim = 64
        self.bi = True
        self.batch_size = 300
        self.repeat_period = 15
        self.voc_size = 49
        self.embedding_dim = 16
        self.max_step = 200

        self.seq_length = 1024

        self.cuda = False
        if torch.cuda.device_count() > 0:
            print('cuda detected!')
            self.cuda = True


cfg = Config()
