import torch


class Config(object):
    def __init__(self):
        self.hidden_dim = 16
        self.bi = True
        self.batch_size = 10
        self.repeat_period = 10
        self.voc_size = 25
        self.embedding_dim = 16
        self.max_step = 300

        self.seq_length = 1024

        self.cuda = False
        if torch.cuda.device_count() > 0:
            print('cuda detected!')
            self.cuda = True


cfg = Config()
