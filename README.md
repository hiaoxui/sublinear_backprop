# Sublinear Backpropagation

## Done:

* Coding
* Comments (Hope someone could help me to fix grammar bugs)
* Test for correctness

## To Do:

* Evaluate the additional computational cost
* Optimize

## Experiment on time overhead:

Settings: A silly task, with my poor GeForce 940MX. RNNs Cell: torch.nn.LSTMCell

* Sequence Length: 4096
* Vocabulary Size: 25
* Embedding Dim: 64
* Hidden Dim: 40
* Batch Size: 50

Uni-directional:

| n_chunk | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 | 256 | 512 | 1024 | 2048 | 4096 |
| --- | ---| --- | --- | --- | ---| --- | --- | --- | ---| --- | --- | --- | --- |
| t_inf | 0.371 | 0.340 | 0.355 | 0.360 | 0.345 | 0.348 | 0.352 | 0.353 | 0.388 | 0.432 | 0.520 | 0.679 | 0.970 |
| t_est | 26.3 | 14.2 | 7.74 | 4.81 | 3.24 | 2.59 | 2.40 | 2.47 | 2.75 | 3.21 | 4.14 | 5.44 | 8.29 |

Gold (cuDNN): t_inf = 0.028, t_est = 0.387

Bi-directional:

| n_chunk | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 | 256 | 512 | 1024 | 2048 | 4096 |
| --- | ---| --- | --- | --- | ---| --- | --- | --- | ---| --- | --- | --- | --- |
| t_inf | 0.843 | 0.935 | 1.14 | 1.30 | 1.46 | 1.62 | 1.76 | 1.92 | 2.18 | 2.52 | 2.93 | 3.74 | 5.13 |
| t_est | 53.5 | 28.7 | 16.2 | 10.1 | 7.11 | 5.71 | 5.24 | 5.33 | 6.04 | 7.08 | 8.55 | 11.6 | 17.2 |

Gold (cuDNN): t_inf = 0.0407, t_est = 0.590
