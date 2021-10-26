# PPG DaLiA Dataset
## Download 
The dataset is freely available at https://archive.ics.uci.edu/ml/datasets/PPG-DaLiA. 
Before running experiments on this dataset please download the data.

## Overview
PPG-DaLiA is a publicly available dataset for PPG-based heart rate estimation. This multimodal dataset features physiological and motion data, recorded from both a wrist- and a chest-worn device, of 15 subjects while performing a wide range of activities under close to real-life conditions. The included ECG data provides heart rate ground truth. The included PPG- and 3D-accelerometer data can be used for heart rate estimation, while compensating for motion artefacts. Further details can be found in the dataset's readme-file.

### Data pre-processing and loading
The data pre-processing part is managed with the function defined in **preprocessing/preprocessing_Dalia.py** where the scheme originally proposed in [1] is followed.

---
[1]: Attila Reiss, Ina Indlekofer, Philip Schmidt, and Kristof Van Laerhoven. 2019. Deep PPG: Large-scale Heart Rate Estimation with Convolutional Neural Networks. MDPI Sensors, 19(14).