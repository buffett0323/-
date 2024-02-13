# README

資料集：原始資料集做Sequence、前處理<br>
結合國土利用資料kmz2000(seq_processing)<br>
結合遙測集資料JAXA_HRLULC_Japan_v23.12<br>
將行政區level 1, 2資料gadm41帶入<br>

資料集樣貌newDataShift_{SEQ_LENGTH}：<br>

0. time
1. gender (static)
2. age (static)
3. work (static)
4. trip purpose
5. transport type
6. longitude
7. latitude
8. 國土利用 https://www.chikyu.ac.jp/USE/navi9-e.html
9.  遙測資料 // hour
---------------------
10. level 1 y label
11. level 2 y label


LSTM, hidden layers: 128, learning rate: 0.01, 82.96% <br>
LSTM, hidden layers: 64, learning rate: 0.01, 81.68% <br>
GRU, hidden layers: 64, learning rate: 0.01, 79.25% <br>

Seq_4: 84.28%
Seq_8: 82.41%

Hybrid LSTM Accuracy: Layer = 2
| lr, fc | 32 | 64 | 128 | 256 |
|----|-------|-----|----|----|
| 0.001 | 81.42% | 81.83% | 82.5% | 82.7% |
| 0.005 | 83.81% | 84.08% | 83.72% | 83.43% |
| 0.01 | 83.84% | 83.73% | 83.7% | 83.25% |

Layer = 3: 84.26%
One hot encoding: bad
有 l2_lambda 比較高


### Hybrid Weighted Linear
| Weights | 2 | 3 | 4 |
|----|----|----|----|
| Linear | 81.68% | 81.58% | 81.38% |


### Hybrid Weighted Exponential
| Weights | 4/3 | 3/2 | 2 |
|----|----|----|----|
| Linear | 79.06% | 80.38% | 80.08% |


### Different Sequence on Hybrid Model
| Sequence | 4 | 5 | 6 | 7 | 8 | 9 |
| ----- | ----- |  ----- |  ----- |  ----- |  ----- |  ----- | 
| No BD | 86.77% | 83.11% | 84.11% | 80.98% | 84.45% | 80.58% |
| BD | 84.4% | 81.29% | 82.19% | 79.54% | 82.91% | 79.79% |
* No batchnorm1D, and dropout_rate
Hybrid_LSTM_model_{SEQ}_test.pth
Hybrid_LSTM_model_{SEQ}_new_bd.pth


### CUT 15
normal: 87.54%: LSTM_model_seq_6_cut_15.pth
hybrid without batchnorm1D, dropout: 87.5%: Hybrid_LSTM_model_seq_6_cut_15.pth
hybrid with batchnorm1D, dropout: 87.5%
hybrid without bd: 87.03%, hybrid with bd: 87.33%, using batch and dropout is better.


### 0208
Seq = 4, batch = True
Stored in new_model_pth/

lr = 0.005
| hl, fc | 64 | 128 | 256 |
|----|-------|-----|----|
| 64 | 0% | 0% | 0% | 
| 128 | 0% | 85.58 % | 85.58 % | 


num_layers = 2, lr = 0.005, no batch, drop_prob = 0
| hl, fc | 64 | 128 | 256 |
|----|-------|-----|----|
| 64 | 86.72 % | 86.69 % | 86.55 % |
| 128 | 86.85 % | 86.74 % | 86.7 % | 

num_layers = 3, fc_layers = 64
| hidden layers | 64 | 128 |
|---|---|---|
| accuracy | 86.91% | 87.11% |

Best: Hybrid_LSTM_model_4_hl_128_fc_64_lr_0.005_nl_3_dr0.pth


### 0212
New model, adding hour and remove 遙測資料
fc_layers = 64, lr = 0.005
| hl, nl | 2 | 3 |
|-------|----|-----|
| 64 | 0% | 87.25% |
| 128 | 86.91% | 87.27% |


### 0213
New Concept: Stay -> Move -> Stay -> Move -> Prediction

hl_32_fc_64_lr_0.005_nl_2: 79.95 %

hl_32_fc_64_lr_0.005_nl_3: 81.42 %

hl_32_fc_64_lr_0.01_nl_2: 81.35 %

hl_32_fc_64_lr_0.01_nl_3: 81.58 %

hl_64_fc_64_lr_0.005_nl_2: 81.43 %
