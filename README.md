## Trip-purpose-based methods for predicting human mobility’s next location
Author: Jeng-Yue Liu


### Different Sequence on Hybrid Model
| Sequence | 3 |4 | 5 | 6 | 7 | 8 | 
| ----- | ----- |  ----- |  ----- |  ----- |  ----- |  ----- | 
| Strict Accuracy | 0.8768 | 0.842 | 0.8599 | 0.8267 |  0.8614 | 0.8181 |
| Adjacent Accuracy | 0.9604 | 0.9482 | 0.9601 | 0.9495 | 0.9665 | 0.9533 | 

### Accuracy of different models 
| Model | Strict accuracy | Adjacent accuracy |
| ---- | ---- | ---- |
| LSTM | 0.8411 | 0.9508 |
| GRU | 0.8428 | 0.9542 |
| Hybrid LSTM | 0.8727 | 0.9602 |
| Hybrid GRU | 0.8768 | 0.9603 |