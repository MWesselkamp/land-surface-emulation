# land-surface-emulation

This repository contains source code for the three land surface emulators introduced in the work "Advances in Land Surface Model-based Forecasting: A comparative study of LSTM, Gradient Boosting, and Feedforward Neural Network Models as prognostic state emulators". (see Arxiv pre-print: https://doi.org/10.48550/arXiv.2407.16463)

Data sources for creating emulators are ERA5 meteorological and physiographic fields (https://rmets.onlinelibrary.wiley.com/doi/10.1002/qj.3803) and simulations from ECMWF's forced land surface scheme EcLand (https://www.mdpi.com/2073-4433/12/6/723).

- src - source code for model development
- configs - files for experiment configurations described in Wesselkamp et al.
- shell_scripts - bash scripts for running experiments on GPU/CPU node of an HPC.

## Source code structure 

- src/data: contains the script with the data module and data_loader.

- src/evaluation: contains the scripts and tools related to evaluating the performance, computing the forecast horizons of the emulators and creating the figures.
  
- src/lstm: contains code related to the implementation of the LSTM, that is neural architecture search, models and training files.

- src/mlp: contains code related to the implementation of the MLP, that is neural architecture search, models and training files.

- src/xgb: contains code related to the implementation of the XGB, that is model training and some helper functions.

- src/utils: contains shared utility functions, that are metrics and visualisation and handling of directories.


## Configuration of experiments

Configuration files contain parameters that specify the experiment on **Input and Target Variables**, **File Path and Data Ranges**, **Data Selection and Preprocessing**, **Transformations**, **Model Training Parameters**, **System parameters**, **Model Selection and Paths**.

### Data selection and Preprocessing

*x_slice_indices* defines the slice indices for spatial dimensions. In this case, 0 to None is selected, which means the entire spatial range is taken. 
*spatial_sample_size* defines a sample size for spatial grid points if data needs to be chunked during the training procedure.
*path_to_boundingbox* is not required in this version.

### Transformations

Four parameters control how data is normalized or transformed before being passed to the model. These are one parameter for each type of input data, i.e. dynamic, static, prognostic or diagnostic. z-scoring is safe to use.

### System parameters

Models are set up to run in a distributed training environment. Specifing *strategy* to ddp chooses the distributed training strategy with Pytorch Lightning used for multiple GPU training. Devices specifies the number of GPUs to be used for training 
