# BDCLDosePrediction
The implementation of the paper "Beam-wise Dose Composition Learning for Head and Neck Cancer Radiotherapy Planning" in Pytorch.
## Performance
* The comparison result of dose distribution map with state-of-the-art methods.
<img src="https://github.com/TL9792/BDCLDosePrediction/blob/main/dosemap.png" width="600px">  

* The quantitative result in terms of two official metrics from the [AAMP OpenKBP-2020 Challenge](https://competitions.codalab.org/competitions/23428#results), i.e., Dose score and DVH score.  

Dose score  |  DVH score    
----  |  ----
2.066±0.900  |  0.977±1.091  

## Requirements  
* torch >= 1.9.1
* numpy
* SimpleITK
* pandas
* os


## Usage  
1. Training and Validation:  
    `python train.py notes`  
    The best training model will be saved in `/result/modelname_notes/models/`  
2. Testing:      
    `python test.py notes`  
    The predicted results will be saved in `/result/modelname_notes/test-results-images/`  
3. Using the pre-trained model:  
   * Download the pre-trained model weight ([Baidu Drive](https://pan.baidu.com/s/1zovDJzHej_akMZy90OgWdQ), PassWord: pvv0), which can be saved in `/bestmodel/`  
   `python test.py notes '/bestmodel/'`  
   The predicted results will be saved in `/result/modelname_notes/test-results-images/`  

## Contact  
Please email us with any questions, email: 1963373350@qq.com.

