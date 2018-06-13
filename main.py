from built_in_lstm import OriginalLSTM
from trainer import Trainer


def main():
    model = OriginalLSTM()
    tr = Trainer(model)
    tr.evaluate(100)
    while True:
        tr.train(1000)
        tr.evaluate(100)


if __name__ == '__main__':
    main()

