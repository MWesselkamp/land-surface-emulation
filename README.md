# Land surface emulation

This repository contains the three land surface emulators introduced in the work "Advances in Land Surface Forecasting: A comparative study of LSTM, Gradient Boosting, and Feedforward Neural Network Models as prognostic state emulators in a case study with EcLand", published at: https://doi.org/10.5194/gmd-18-921-2025. If you use the code, we're happy about a citation.

Data sources for creating emulators are ERA5 meteorological and physiographic fields (https://rmets.onlinelibrary.wiley.com/doi/10.1002/qj.3803) and simulations from ECMWF's forced land surface scheme EcLand (https://www.mdpi.com/2073-4433/12/6/723).

- src - source code for model development
- configs - files for experiment configurations described in Wesselkamp et al.
- shell_scripts - bash scripts for running experiments on GPU/CPU node of HPC.

## Code structure 

- src/data: contains data module and data_loader.

- src/evaluation: contains tools related to evaluating the performance, computing the forecast horizons of the emulators and creating the figures.
  
- src/lstm: contains the LSTM neural architecture search, models and training files.

- src/mlp: contains the MLP neural architecture search, models and training files.

- src/xgb: contains the XGB model training and some helper functions.

- src/utils: contains shared utility functions, metrics and visualisation and handling of directories.


## Configuration of experiments

Configuration files contain parameters that specify the experiment on **Input and Target Variables**, **File Path and Data Ranges**, **Data Selection and Preprocessing**, **Transformations**, **Model Training Parameters**, **System parameters**, **Model Selection and Paths**. Details on the configuration parameters below.

### Data selection and Preprocessing

*x_slice_indices* defines the slice indices for spatial dimensions. In this case, 0 to None is selected, which means the entire spatial range is taken. 
*spatial_sample_size* defines a sample size for spatial grid points if data needs to be chunked during the training procedure.
*path_to_boundingbox* is not required in this version.

### Transformations

Four parameters control how data is normalized or transformed before being passed to the model. These are one parameter for each type of input data, i.e. dynamic, static, prognostic or diagnostic. z-scoring is safe to use.

### System parameters

Models are set up to run in a distributed training environment. Specifing *strategy* to ddp chooses the distributed training strategy with Pytorch Lightning used for multiple GPU training. *devices* specifies the number of GPUs to be used for training. 
