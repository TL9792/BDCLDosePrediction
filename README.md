# BDCLDosePrediction
The implementation of the paper "Beam-wise Dose Composition Learning for Head and Neck Cancer Dose Prediction in Radiotherapy" in Pytorch.
## Performance
* The comparison result of dose distribution map with state-of-the-art methods.
<img src="https://github.com/TL9792/BDCLDosePrediction/blob/main/dosemap.png" width="800px">  

* The quantitative result in terms of two official metrics from the [AAMP OpenKBP-2020 Challenge](https://competitions.codalab.org/competitions/23428#results), i.e., Dose score and DVH score. Our method achieves superior performance compared with other state-of-the-art methods. 
* The reproduction results may have certain deviations due to different GPUs used. 

Dose score  |  DVH score    
----  |  ----
2.066±0.900  |  0.977±1.091  

## Cite
  @article{teng2024beam,
    title={Beam-wise dose composition learning for head and neck cancer dose prediction in radiotherapy},
    author={Teng, Lin and Wang, Bin and Xu, Xuanang and Zhang, Jiadong and Mei, Lanzhuju and Feng, Qianjin and Shen, Dinggang},
    journal={Medical Image Analysis},
    volume={92},
    pages={103045},
    year={2024},
    publisher={Elsevier}
  }

## Contact  
Please email us with any questions for free, email: tenglin2023@shanghaitech.edu.cn.

