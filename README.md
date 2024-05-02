# Trip-purpose-based methods for predicting human mobility’s next location
Author: Jeng-Yue Liu

## Abstract
This study presents an innovative approach to predicting human mobility’s next location, enhancing traditional methodologies with a focus on the shifts in trip purposes within time-series analysis. By integrating static background information, such as sociodemographic data, with geographic land use characteristics, the model effectively differentiates mobility behavior patterns among diverse demographic groups. This integration allows for the accurate capturing of dynamic changes in mobility while preserving the integrity of individual’s background information. The development of a low-complexity hybrid model, which processes both static and dynamic features, further improves the accuracy and adaptability of predictions across various geographical areas.

Employing advanced GeoAI techniques, including **LSTM (Long Short-Term Memory)** and **GRU (Gated Recurrent Unit)** models, the study aligns predictions closely with real-world dynamics and provides valuable insights for urban planning and business strategy formulation. Additionally, the evaluation of prediction performance incorporates not only "Strict Accuracy" but also a novel metric called "Adjacency Accuracy", which accommodates deviations within neighboring ranges.

The model achieves a strict accuracy of **0.7927** and an adjacent accuracy of **0.9199**. This approach promises to offer new perspectives and scientific support for urban economic development, paving the way for further research in applying these methodologies to specific datasets and enhancing urban planning efforts.

### Keywords
- *Next-location prediction*
- *Trip purpose*
- *Human mobility pattern*
- *GeoAI*
- *LSTM*
- *GRU*
- *Deep Learning*

### Overall prediction
This diagram shows the strict accuracy on geographical distribution map in the study:

![Second administrative division prediction](result_visual/visual_0429_4_0429.png "Second administrative division prediction")

