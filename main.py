from built_in_lstm import OriginalLSTM
from trainer import Trainer
import time


def main():
    model = OriginalLSTM()
    tr = Trainer(model)
    since = time.clock()
    tr.evaluate(100)
    print(time.clock() - since)
    while True:
        since = time.clock()
        tr.train(1000)
        print(time.clock() - since)
        tr.evaluate(100)


if __name__ == '__main__':
    main()

