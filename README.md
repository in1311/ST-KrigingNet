# ST-KrigingNet
This is the data and code related to the paper submitted to IJGIS titled 'ST-KrigingNet: Deep Spatiotemporal Kriging Neural Network for Multivariate and Nonstationary Spatiotemporal Interpolation'.

# Requirements
* numpy
* datetime
* os
* pandas
* torch
* tensorboard
* sklearn
* matplotlib
* math
* tqdm

# Running examples
**python main.py**  
You can directly modify the "dataset" variable to use the default parameters under different datasets.
Or you can adjust the specific parameters:
* datafile:  sampled dataset in folders "Data/AirQuality/dataset" and "Data/Temp/dataset"
* lr:  learning rate
* hidden_neuron:  [input dimension, model dimension, trend dimension]. Note that the input dimension should be equal to the number of all variables (auxiliary and target) in the dataset
* weight:  weight of position embedding and time embedding 
* batch_size: batch size
* loss_type:  loss function type
* optim_type:  optimizer type
* if_summary:  if save the training summary or not
* if_save_model:  if save the best model or not
* info: extra information for saving folder naming only


The train log and results are saved in folder "results"
