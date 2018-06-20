# Sublinear Backpropagation

## Done:

* Coding
* Comments (Hope someone could help me to fix grammar bugs)
* Test for correctness

## To Do:

* Evaluate the additional computational cost
* Optimize

## Experiment on time and memory overhead:

Settings: A silly task, with my poor GeForce 940MX. RNNs Cell: torch.nn.LSTMCell

* Sequence Length: 4096
* Vocabulary Size: 25
* Embedding Dim: 64
* Hidden Dim: 40
* Batch Size: 50

Uni-directional:

| n_chunk | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 | 256 | 512 | 1024 | 2048 | 4096 |
| --- | ---| --- | --- | --- | ---| --- | --- | --- | ---| --- | --- | --- | --- |
| t_inf/s | 0.387 | 0.340 | 0.355 | 0.360 | 0.345 | 0.348 | 0.352 | 0.353 | 0.388 | 0.432 | 0.520 | 0.679 | 0.970 |
| t_est/s | 26.3 | 14.2 | 7.74 | 4.81 | 3.24 | 2.59 | 2.40 | 2.47 | 2.75 | 3.21 | 4.14 | 5.44 | 8.29 |
| m_inf/MB | 459 | 238 | 124 | 78.7 | 36.6 | 24.3 | 16.3 |13.9  | 11.7 | 10.6 | 10.3  | 11.7 | 12.3  |
| m_est/MB | 474 | 550  | 288 | 160 | 89.2 | 55.3 | 30.5 | 21.7 | 17.3 | 14.4 | 13.2 | 15.2 | 19.1 |

Gold (cuDNN): t_inf = 0.18, t_est = 0.38

Bi-directional:

| n_chunk | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 | 256 | 512 | 1024 | 2048 | 4096 |
| --- | ---| --- | --- | --- | ---| --- | --- | --- | ---| --- | --- | --- | --- |
| t_inf/s | 0.843 | 0.935 | 1.14 | 1.30 | 1.46 | 1.62 | 1.76 | 1.92 | 2.18 | 2.52 | 2.93 | 3.74 | 5.13 |
| t_est/s | 53.5 | 28.7 | 16.2 | 10.1 | 7.11 | 5.71 | 5.24 | 5.33 | 6.04 | 7.08 | 8.55 | 11.6 | 17.2 |
| m_inf/MB | 989 | 696 | 354 | 174 | 93.6 | 47.4 | 32.0 | 21.2 | 15.8 | 12.3 | 13.0 | 15.4 | 21.8 |
| m_est/MB | 970 | 545 | 397 | 163 | 97.5 | 54.5 | 29.5 | 21.6 | 18.7 | 15.8 | 16.0 | 18.7 | 25.5|

Gold (cuDNN): t_inf = 0.0407, t_est = 0.590
