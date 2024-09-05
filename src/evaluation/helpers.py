import sys 
import os
import torch 
import yaml
import pandas as pd
import numpy as np
import xgboost as xgb

from torch import tensor
from utils.visualise import *

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
print(SCRIPT_DIR)

if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
DEVICE = dev
print(DEVICE)

def plot_losses_and_metrics(losses, config):
    """Plot various loss metrics."""
    plot_losses(losses, which='loss_logit', save_to=config['model_path'])
    plot_losses_targetwise(losses, save_to=config['model_path'], label="SmoothL1")
    plot_losses_targetwise_boxplots(losses, save_to=config['model_path'], label="SmoothL1", log=False)


def load_config(configs_path, config_file):
    """Load configuration from a YAML file."""
    config_path = os.path.join(configs_path, config_file)
    try:
        with open(config_path) as stream:
            config = yaml.safe_load(stream)
            print(f"Opening {config_file} for experiment configuration.")
            return config
    except yaml.YAMLError as exc:
        print(exc)
        return None
        
def process_chunkwise(dataset, config, lstm_module, chunk_size):
    """Process data in chunks and run forecasts."""
    total_size = dataset.x_size
    print("TOTAL DATA SIZE:", total_size)
    num_chunks = (total_size + chunk_size - 1) // chunk_size 
    print("SPLIT DATA IN CHUNKS:", num_chunks)

    lstm_prog_chunks = []
    lstm_prog_prediction_chunks = []

    for i in range(num_chunks):
        
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, total_size)
        print(f"CHUNK start: {start_idx}, CHUNK end: {end_idx}")

        config_temp = config.copy()
        config_temp["x_slice_indices"] = (start_idx, end_idx)

        dataset_temp = EcDataset(config_temp, config_temp['test_start'], config_temp['test_end'])
        X_static, X_met, Y_prog = lstm_module.get_test_data(dataset_temp, chunk_idx=(start_idx, end_idx))

        lstm_prog_chunk, lstm_prog_prediction_chunk = lstm_module.step_forecast(X_static, X_met, Y_prog)
        lstm_prog_chunks.append(lstm_prog_chunk)
        lstm_prog_prediction_chunks.append(lstm_prog_prediction_chunk)

    lstm_prog = np.concatenate(lstm_prog_chunks, axis=1)
    lstm_prog_prediction = np.concatenate(lstm_prog_prediction_chunks, axis=1)
    return lstm_prog, lstm_prog_prediction

def process_full_dataset(dataset, lstm_module):
    """Process the entire dataset at once and run forecasts."""
    X_static, X_met, Y_prog = lstm_module.get_test_data(dataset)
    print(f"X_static: {X_static.shape}, X_met: {X_met.shape}, Y_prog: {Y_prog.shape}")
    return lstm_module.step_forecast(X_static, X_met, Y_prog)

def print_best_and_worst_gridcells(lstm_performance_total):
    
    """Print the best and worst grid cells based on performance metrics."""
    print("Worst grid cell in RMSE:", np.argmax(np.array(lstm_performance_total['rmse'])))
    print("Best grid cell in RMSE:", np.argmin(np.array(lstm_performance_total['rmse'])))
    print("Worst grid cell in R2:", np.argmin(np.array(lstm_performance_total['r2'])))
    print("Best grid cell in R2:", np.argmax(np.array(lstm_performance_total['r2'])))


def load_model_from_checkpoint(path_to_results, modelname = 'lstm', my_device = None):

    if my_device is None:
        my_device = DEVICE
    else:
        my_device = my_device
            
    path_to_metrics = os.path.join(path_to_results, 'metrics.csv')
    path_to_checkpoint = os.path.join(path_to_results, 'checkpoints')

    print("Running forecast for model:")

    use_checkpoint = os.listdir(path_to_checkpoint)[-1]
    path_to_best_checkpoint = os.path.join(path_to_checkpoint, use_checkpoint)  # trainer.checkpoint_callback.best_model_path
    print('Path to best checkpoint:', path_to_best_checkpoint)

    checkpoint = torch.load(path_to_best_checkpoint, map_location=torch.device(my_device) )
    
    with open(os.path.join(path_to_results, "hparams.yaml")) as stream:
        try:
            hpars = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            hpars = checkpoint['hyper_parameters']
            
    print("Hyper parameters from Checkpoint: ", hpars)
    if hpars is None:
        hpars = checkpoint['hyper_parameters']
    print("Hyper parameters from Checkpoint: ", hpars)
    
    model_weights = checkpoint['state_dict']
    
    if 'mlp_2D' in path_to_results:
        
        if ('mlp' == modelname):
            from mlp_2D.models import MLPregressor
            print("Set up model: mlp_2D")
            
            model = MLPregressor(input_size=hpars['input_size'], 
                             hidden_size=hpars['hidden_size'], 
                             output_size = hpars['output_size'], 
                             batch_size=None, 
                             learning_rate= hpars['learning_rate'], 
                             weight_decay = hpars['weight_decay'], 
                            lookback = hpars['lookback'], 
                             device = my_device, 
                             rollout=hpars['rollout'], 
                             dropout=hpars['dropout'])
            
        elif ('mlp_global' == modelname):
            from mlp_2D.models import MLPregressor_global as MLPregressor
            print("Set up model: mlp_2D global")
            
            model = MLPregressor(input_clim_dim = hpars['input_clim_dim'],
                             input_met_dim = hpars['input_met_dim'],
                             input_state_dim = hpars['input_state_dim'],
                             hidden_dim = hpars['hidden_dim'],
                             output_dim = hpars['output_dim'],
                             output_diag_dim = hpars['output_diag_dim'],
                             batch_size=hpars["batch_size"],
                             learning_rate=hpars['learning_rate'], 
                             lookback = hpars['lookback'], 
                             rollout=hpars["rollout"], 
                             dropout=hpars['dropout'], 
                             weight_decay = hpars['weight_decay'], 
                             loss = hpars['loss'],
                             device = my_device, 
                             targets = hpars["targets"],
                             db_path = hpars["db_path"])
        
    elif ('lstm_2D' in path_to_results) | ('lstm' == modelname) | ('leadtime' in modelname) :
        
        if ('lstm_statesemb_logtransform' == modelname):
            from lstm_2D.models import LSTM_m2m_statesemb as LSTMregressor
        elif ('lstm_autoencoder' == modelname):
            from lstm_2D.models import LSTM_m2m_autoencoder as LSTMregressor
        elif ('lstm_autoencoder_nc' == modelname):
            from lstm_2D.models import LSTM_m2m_autoencoder_nc as LSTMregressor
        elif ('lstm_global' == modelname):
            from lstm_2D.models import LSTM_m2m_global as LSTMregressor
        elif ('leadtime' in modelname):
            from lstm_2D.models import LSTM_m2m_autoencoder_nc as LSTMregressor
        else:
            print("Specify which model to load!")
            print("Choose from: lstm_basic,lstm_basic_usedlogits,lstm_statesemb, lstm_projector")
            
        print("Set up model: lstm_2D")
        if modelname == 'lstm_global':
            
            model = LSTMregressor(input_clim_dim = hpars['input_clim_dim'] ,
                                 input_met_dim = hpars['input_met_dim'],
                                 input_state_dim = hpars['input_state_dim'],
                                 spatial_points = hpars['spatial_points'], 
                                  lookback = hpars['lookback'], 
                                  lead_time = hpars['lead_time'], 
                                  num_layers_en=hpars['num_layers_en'],
                                  num_layers_de=hpars['num_layers_de'],
                                  hidden_size_en = hpars['hidden_size_en'],
                                  hidden_size_de = hpars['hidden_size_de'],
                                  learning_rate = hpars['learning_rate'],
                                  weight_decay = hpars['weight_decay'],
                                  use_dlogits =  hpars['use_dlogits'],
                                  db_path = hpars['file_path'],
                                  device = my_device,
                                  batch_size=1, # set to one for forecasting!
                                  dropout = 0)
        else:
            model = LSTMregressor(input_features = hpars['input_features'], 
                                  lookback = hpars['lookback'], 
                                  lead_time = hpars['lead_time'], 
                                  spatial_points= hpars['spatial_points'], #hpars['spatial_points'], #1000
                                  num_layers_en=hpars['num_layers_en'],
                                  num_layers_de=hpars['num_layers_de'],
                                  hidden_size_en = hpars['hidden_size_en'],
                                  hidden_size_de = hpars['hidden_size_de'],
                                  learning_rate = hpars['learning_rate'],
                                  weight_decay = hpars['weight_decay'],
                                  use_dlogits =  hpars['use_dlogits'],
                                  device = my_device,
                                  batch_size=1, # set to one for forecasting!
                                  dropout = 0)
    
    else:
        print("Don't know model type!")
        
    # forecast precision high
    torch.set_float32_matmul_precision("high")

    model.load_state_dict(model_weights)
    print("Now returning model:", model)
    
    return model

def remove_first_directory(path):
    parts = path.split(os.sep)  # Split the path into parts based on the OS-specific separator
    if parts and parts[0] == '':  # Handle absolute paths
        parts = parts[1:]  # Remove the leading empty string for absolute paths
    new_path = os.path.join("../", os.sep.join(parts[1:])) # Join the parts back together, skipping the first directory
    return new_path

def load_parameters(model_path, my_device):
    
    """this functions loads hyperparameters from accompaning yaml file if possible, otherwise from checkpoint directly!"""

    checkpoint = torch.load(os.path.join(model_path, "checkpoints/last.ckpt"), 
                                map_location=torch.device(my_device))
        
    with open(os.path.join(model_path, "hparams.yaml")) as stream:
        try:
            hpars = yaml.safe_load(stream)
            print("Hyper parameters from YAML: ", hpars)
        except yaml.YAMLError as exc:
            print(exc)
            hpars = checkpoint['hyper_parameters']
            print("Hyper parameters from Checkpoint: ", hpars)
        
    model_weights = checkpoint['state_dict']
    for key in model_weights.keys():
        print(key)

    return hpars, model_weights


def load_model_with_config(config, my_device = None):

    if my_device is None:
        device = DEVICE
    else:
        device = my_device

    my_device = device if config["device"] is None else config["device"]
    print("Set DEVICE to:", my_device)
    
    if config['model'] != 'xgb':

        # Create exception for working with different filepath spec in notebooks
        try:
            hpars, model_weights = load_parameters(config['model_path'], my_device)
        except FileNotFoundError:
            hpars, model_weights = load_parameters(remove_first_directory(config['model_path']), my_device)
        
    if config['model'] == 'mlp':
    
        if ('global' in config['logging']['name']) | ('europe' in config['logging']['name']):

            from mlp_2D.models import MLPregressor_global as MLPregressor
            print("Set up model: mlp_2D global")

            model = MLPregressor(input_clim_dim = hpars['input_clim_dim'],
                                 input_met_dim = hpars['input_met_dim'],
                                 input_state_dim = hpars['input_state_dim'],
                                 hidden_dim = hpars['hidden_dim'],
                                 output_dim = hpars['output_dim'],
                                 output_diag_dim = hpars['output_diag_dim'],
                                 batch_size=hpars["batch_size"],
                                 learning_rate=hpars['learning_rate'], 
                                 lookback = config['lookback'], 
                                 rollout=config["roll_out"], 
                                 dropout=hpars['dropout'], 
                                 weight_decay = hpars['weight_decay'], 
                                 loss = hpars['loss'],
                                 device = my_device, 
                                 targets = config["targets_prog"],
                                 db_path = config["file_path"])


        else:
            from mlp_2D.models import MLPregressor
            print("Set up model: mlp_2D")

            model = MLPregressor(input_size=hpars['input_size'], 
                                 hidden_size=hpars['hidden_size'], 
                                 output_size = hpars['output_size'], 
                                 batch_size=None, 
                                 learning_rate= hpars['learning_rate'], 
                                 weight_decay = hpars['weight_decay'], 
                                lookback = hpars['lookback'], 
                                 device = my_device, 
                                 rollout=hpars['rollout'], 
                                 dropout=hpars['dropout'])
        torch.set_float32_matmul_precision("high")
    
        model.load_state_dict(model_weights)
            
    elif config['model'] == 'lstm':
        
        if ('fieldsemb' in config['logging']['name']):
            from lstm_2D.models import LSTM_m2m_global_fieldsemb as LSTMregressor
            print("Set up model: lstm_2D global fieldsemb")
        else:
            from lstm_2D.models import LSTM_m2m_global as LSTMregressor
            print("Set up model: lstm_2D global")
            
        model = LSTMregressor(input_clim_dim = hpars['input_clim_dim'] ,
                                 input_met_dim = hpars['input_met_dim'],
                                 input_state_dim = hpars['input_state_dim'],
                                 spatial_points = hpars['spatial_points'], 
                                  lookback = config['lookback'], 
                                  lead_time = config['roll_out'], 
                                  num_layers_en=hpars['num_layers_en'],
                                  num_layers_de=hpars['num_layers_de'],
                                  hidden_size_en = hpars['hidden_size_en'],
                                  hidden_size_de = hpars['hidden_size_de'],
                                  learning_rate = hpars['learning_rate'],
                                  weight_decay = hpars['weight_decay'],
                                  use_dlogits =  hpars['use_dlogits'],
                                  db_path = config['file_path'],
                                  device = my_device,
                                  batch_size=1, # set to one for forecasting!
                                  dropout = 0)

        for key in model.state_dict().keys():
            print(key)

        torch.set_float32_matmul_precision("high")
    
        model.load_state_dict(model_weights)

    elif config['model'] == 'xgb':

        print("Set up model: XGB")
        
        model = xgb.Booster()
        model.load_model(os.path.join(config['model_path'], 'xgb_model.bin'))
        
    else:
        print("SPECIFY which model type to load")

    print("Now returning model type:", model)
    
    return model