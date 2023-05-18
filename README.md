# Imputation-based Time-Series Anomaly Detection with Conditional Weight-Incremental Diffusion Models

This is an repository hosting the code of DiffAD.

## Datasets

1. PSM (PooledServer Metrics) is collected internally from multiple application server nodes at eBay.
   You can learn about it
   from [Practical Approach to Asynchronous Multivariate Time Series Anomaly Detection and Localization](https://dl.acm.org/doi/abs/10.1145/3447548.3467174)
   .
2. MSL (Mars Science Laboratory rover) is a public dataset from NASA. You can learn about it 
   from [Detecting Spacecraft Anomalies Using LSTMs and Nonparametric Dynamic Thresholding](https://arxiv.org/pdf/1802.04431.pdf).
3. SMAP (Soil Moisture Active Passive satellite) also is a public dataset from NASA. You can learn about it
   from [Detecting Spacecraft Anomalies Using LSTMs and Nonparametric Dynamic Thresholding](https://arxiv.org/pdf/1802.04431.pdf).
4. SMD (Server Machine Dataset) is a 5-week-long dataset collected from a large Internet company. You can learn about it
   from [Robust Anomaly Detection for Multivariate Time Series through Stochastic Recurrent Neural Network
   ](https://netman.aiops.org/wp-content/uploads/2019/08/OmniAnomaly_camera-ready.pdf).
5. SWaT (Secure Water Treatment) is obtained from 51 sensors of the critical infrastructure system under continuous
   operations. You can learn about it from [SWaT: a water treatment testbed for research and training on ICS security
   ](https://ieeexplore.ieee.org/abstract/document/7469060).

## Usage

### Environment

Install Python 3.8.

```python
pip install -r requirements.txt
```

By default, datasets are placed under the "tf_dataset" folder. If you need to change 
the dataset, you can modify the dataset path  in the json file in the "config" folder. 
Here is an example of modifying the training dataset path:

```json
"datasets": {
    "train|test": {
        "dataroot": "tf_dataset/smap/smap_train.csv",
        //"dataroot": "tf_dataset/swat/swat_train.csv"
    }
},
```
In addition, we provide json configuration files 
for two datasets (SMAP and PSM) for reference.

### Training
Next, we demonstrate using the SMAP dataset.

#### We use dataset SMAP for training demonstration.

```python
# Use time_train.py to train the task.
# Edit json files to adjust dataset path, network structure and hyperparameters.
python time_train.py -c config/smap_time_train.json
```

### Test
The trained model is placed in "experiments/*/checkpoint/" by default. 
If you need to modify this path, you can refer to "config/smap_time_test.json":

```json
"path": {
  "resume_state": "experiments/SMAP_TRAIN_128_2048_100/checkpoint/E100"
},
```
 
#### We also use dataset SMAP for testing demonstration.

```python
# Edit json to adjust pretrain model path and dataset_path.
python time_test.py -c config/smap_time_test.json
```

#### RESULT
The GPU we use is NVIDIA RTX3090 24GB, the training time is about 1 hour, 
and the test time is about half an hour. 
The following is the F1-score obtained after testing the SMAP dataset.
<p align="center">
<img src=".\pics\result.png" width="500" height = "130" alt="result" align=center />
</p>
