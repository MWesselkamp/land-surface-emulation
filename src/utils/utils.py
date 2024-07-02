import numpy as np
import os
import pandas as pd
import pytorch_lightning as pl
import torch
import random
import xarray as xr
import re
import xgboost as xgb


from scipy.ndimage import shift
from datetime import datetime
from sklearn.metrics import r2_score, mean_absolute_percentage_error, mean_absolute_error, mean_squared_error


def create_folder_datetime(directory_path):
    """
    Create a new experiment folder with a timestamp as the name within the specified directory path.

    Parameters:
    - directory_path (str): The path to the directory where the experiment folder will be created.

    Returns:
    - str: The path to the newly created experiment folder.

    Raises:
    - Exception: If an error occurs while creating the folder.
    """
    try:
        # Get the current date and time
        current_datetime = datetime.now()
        # Generate a timestamp in the "yymmdd_hhss" format
        timestamp = current_datetime.strftime("%y%m%d_%H%M")
        # Create a new folder with the timestamp as the name
        new_folder_name = f"version_{timestamp}"
        new_folder_path = os.path.join(directory_path, new_folder_name)

        if not os.path.exists(new_folder_path):
            os.makedirs(new_folder_path)
            print(f"Created folder: {new_folder_path}")
        else:
            print(f"Folder already exists: {new_folder_path}")

        return new_folder_path  # Return the path to the new folder

    except Exception as e:
        print(f"An error occurred: {str(e)}")  # Handle any exceptions and print an error message

def create_folder_manually(directory_path, new_folder_name):
    """
    Create a new scenario folder with the specified name within the specified directory path.

    Parameters:
    - directory_path (str): The path to the directory where the scenario folder will be created.
    - new_folder_name (str): The name of the new scenario folder to be created.

    Returns:
    - str: The path to the newly created scenario folder.

    Raises:
    - Exception: If an error occurs while creating the folder.
    """
    try:
        new_folder_path = os.path.join(directory_path, new_folder_name)

        if not os.path.exists(new_folder_path):
            os.makedirs(new_folder_path)
            print(f"Created folder: {new_folder_path}")
        else:
            print(f"Folder already exists: {new_folder_path}")

        return new_folder_path

    except Exception as e:
        print(f"An error occurred: {str(e)}")

def encode_doy(doy):
    """Encode the day of the year on circular coordinates.
    Thanks to: Philipp Jund.
    """
    doy_norm = doy / 365 * 2 * np.pi

    return np.sin(doy_norm), np.cos(doy_norm)



def inverse_variance(yosm):
    """
    Takes a time series and computes the inverse variance of the temporal changing rate.
    :param yosm: Input observations, matrix. Can be taken from datat loader such as dm.y_train[params['seq_len']:, :]
    :return: vector. Inverse variance for each variable column in yosm.
    """
    yosm_n = shift(yosm, 1)
    inv_var = np.var(np.subtract(yosm_n, yosm), axis=0) ** (-1)

    return inv_var


def get_hp_search_results(parent_folder, column_name, file_prefix='', file_extension='.csv'):
    """
    Extracts a specific column from CSV files in numbered subfolders within a parent folder.

    Args:
        parent_folder (str): The path to the parent folder containing subfolders.
        column_name (str): The name of the column to extract.
        file_prefix (str, optional): Prefix of the CSV files (if needed).
        file_extension (str, optional): File extension of the CSV files (default is '.csv').

    Returns:
        list: A list of column values from all CSV files.

    # Example usage:
    parent_folder = 'experiments/opt_lstm/lightning_logs'
    desired_column_name = 'val_acc'
    results = extract_column_from_csv_files(parent_folder, desired_column_name)

    minimum = min(results.values())
    position = [key for key, value in results.items() if value == minimum]
    """
    column_values = []
    version = []
    # Iterate over subfolders in the parent folder
    for subfolder_name in os.listdir(parent_folder):
        version.append(subfolder_name)
        subfolder_path = os.path.join(parent_folder, subfolder_name)

        # Check if the subfolder is a directory
        if os.path.isdir(subfolder_path):
            # Iterate over files in the subfolder
            for filename in os.listdir(subfolder_path):
                if filename.startswith(file_prefix) and filename.endswith(file_extension):
                    csv_file_path = os.path.join(subfolder_path, filename)
                    # Read the CSV file using Pandas
                    df = pd.read_csv(csv_file_path)

                    # Check if the specified column exists in the DataFrame
                    if column_name in df.columns:
                        column_values.append(df[column_name].dropna().tolist()[-1])

    results = dict(zip(version, column_values))

    return results

def best_trial(results, column_name):

    minimum = min(results.values())
    position = [key for key, value in results.items() if value == minimum]
    print(f"Best trial is {position} with {column_name}: {minimum}")

    return position[0]

def get_latest_version_path(folder_path):
    subfolders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]

    if not subfolders:
        print("COULDNT FIND A VERSION FOLDER")  # No subfolders found
    else:
        print(f"FOUND {len(subfolders)} VERSION FOLDERS")

    try:
        latest = max([int(list(filter(str.isdigit, subfolder))[0]) for subfolder in subfolders])
        largest_index_subfolder = os.path.join(folder_path, f'version_{latest}')
        return largest_index_subfolder

    except ValueError:
        print("COULDNT FIND A VERSION")  # Ignore non-integer subfolder names

def next_version_folder_name(parent_dir):
    # Regular expression to match directories like version_0, version_1, etc.
    pattern = re.compile(r'^version_(\d+)$')

    # List all items in the parent directory
    items = os.listdir(parent_dir)
    
    # Filter and sort the version directories based on the number
    version_dirs = sorted(
        [item for item in items if os.path.isdir(os.path.join(parent_dir, item)) and pattern.match(item)],
        key=lambda x: int(pattern.match(x).group(1))
    )
    
    # Find the latest version number
    if version_dirs:
        latest_version = int(pattern.match(version_dirs[-1]).group(1))
    else:
        latest_version = -1  # If no such directories exist, start with version_0
    
    # Create a new directory with version_n+1
    new_version_dir = f"version_{latest_version + 1}"
    #new_version_path = os.path.join(parent_dir, new_version_dir)
    #os.makedirs(new_version_path, exist_ok=True)
    
    print(f"New folder name: {new_version_dir}")
    
    return new_version_dir

def r2_score_multi(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Calculated the r-squared score between 2 arrays of values

    :param y_pred: predicted array
    :param y_true: "truth" array
    :return: r-squared metric
    """
    return r2_score(y_pred.flatten(), y_true.flatten())


def extract_and_print_dimensions(dataset, idx):
    # Extract the data for the given index
    X_h_combined, Y_h, X_f_combined, Y_f = dataset[idx]

    # Print the dimensions
    print("X_h_combined dimensions:", X_h_combined.shape)
    print("Y_h dimensions:", Y_h.shape)
    print("X_f_combined dimensions:", X_f_combined.shape)
    print("Y_f dimensions:", Y_f.shape)

    X_h_combined = X_h_combined.unsqueeze(0)
    X_f_combined = X_f_combined.unsqueeze(0)
    # Return the extracted data in case you need to use it further
    return X_h_combined, Y_h, X_f_combined, Y_f


def seed_everything(seed):
    random.seed(seed)  # Python's built-in random module
    np.random.seed(seed)  # Numpy module
    torch.manual_seed(seed)  # PyTorch random number generator for CPU
    torch.cuda.manual_seed_all(seed)  # PyTorch random number generator for all GPUs
    pl.seed_everything(seed, workers=True)  # PyTorch Lightning utility that covers the above and more
    
    # Configure PyTorch for deterministic behavior (might reduce performance)
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = False

    
def debug_lstm(X_h, hidden_cell=None):
    import torch.nn as nn
    import torch
    # Check input shape
    print("Input Shape:", X_h.shape)
    lstm_encoder = nn.LSTM(input_size=31 * 10,
                           hidden_size=7 * 10,
                           num_layers=1,
                           dropout=0.2,
                           batch_first=True)
    # If hidden_cell is not provided, initialize it automatically
    if hidden_cell is None:
        num_layers = lstm_encoder.num_layers
        hidden_size = lstm_encoder.hidden_size
        batch_size = X_h.size(0)  # Assuming X_h has shape (batch, seq_len, input_size)

        # Initialize hidden and cell states
        hidden_cell = (torch.zeros(num_layers, batch_size, hidden_size),
                       torch.zeros(num_layers, batch_size, hidden_size))

    # Pass the input through the lstm_encoder
    out, (sn, cn) = lstm_encoder(X_h, hidden_cell)

    # Inspect output shapes
    print("Output Shape:", out.shape)
    print("Final Hidden State Shape:", sn.shape)
    print("Final Cell State Shape:", cn.shape)
    
    
def get_scores_spatial(preds, obs, dataset, targetwise = True, save_to=''):
    
    targ_lst = dataset.targ_lst
    
    print("calculating total scores")
    
    rmses = [mean_squared_error(preds[:,p, -len(targ_lst):], obs[:, p,-len(targ_lst):], squared=False) for p in range(obs.shape[1])]
    maes = [mean_absolute_error(preds[:,p, -len(targ_lst):], obs[:, p,-len(targ_lst):]) for p in range(obs.shape[1])]
    r2 = [r2_score_multi(preds[:,p, -len(targ_lst):], obs[:, p,-len(targ_lst):]) for p in range(obs.shape[1])]

    print(len(rmses))
    print(len(dataset.ds_ecland['x']))
    
    performance_total = pd.DataFrame()
    performance_total['x'] = dataset.ds_ecland['x']
    performance_total['lat'] = dataset.ds_ecland['lat']
    performance_total['lon'] = dataset.ds_ecland['lon']
    performance_total['rmse'] = rmses
    performance_total['mae'] = maes
    performance_total['r2'] = r2
    if save_to is not None:
        performance_total.to_csv(os.path.join(save_to, f'scores_spatial_total.csv'))

    performance_targetwise = {}
    
    if targetwise:
        
        for targ in range(len(targ_lst)):
            
            print("calculating target scores:", targ_lst[targ])

            rmses = [mean_squared_error(preds[:,p, -(len(targ_lst))+targ], obs[:, p,-(len(targ_lst))+targ], squared=False) for p in range(obs.shape[1])]
            maes = [mean_absolute_error(preds[:,p, -(len(targ_lst))+targ], obs[:, p,-(len(targ_lst))+targ]) for p in range(obs.shape[1])]
            r2 = [r2_score(preds[:,p, -(len(targ_lst))+targ], obs[:, p,-(len(targ_lst))+targ]) for p in range(obs.shape[1])]

            performance = pd.DataFrame()
            performance['x'] = dataset.ds_ecland['x']
            performance['lat'] = dataset.ds_ecland['lat']
            performance['lon'] = dataset.ds_ecland['lon']
            performance['rmse'] = rmses
            performance['mae'] = maes
            performance['r2'] = r2
            if save_to is not None:
                performance.to_csv(os.path.join(save_to, f'scores_spatial_{targ_lst[targ]}.csv'))

            performance_targetwise[targ_lst[targ]] = performance
    
    return performance_total, performance_targetwise

def get_scores_spatial_global(preds, obs, dataset, clim = None, rolling = 10, targetwise = True, save_to=''):
    
    targ_lst = dataset.targ_lst
    
    print("calculating total scores")
    
    rmses = [mean_squared_error(preds[:,p, -len(targ_lst):], obs[:, p,-len(targ_lst):], squared=False) for p in range(obs.shape[1])]
    maes = [mean_absolute_error(preds[:,p, -len(targ_lst):], obs[:, p,-len(targ_lst):]) for p in range(obs.shape[1])]
    r2 = [r2_score_multi(preds[:,p, -len(targ_lst):], obs[:, p,-len(targ_lst):]) for p in range(obs.shape[1])]

    if clim is not None:
        acc = [anomaly_correlation(preds[:, p, -len(targ_lst):], obs[:, p, -len(targ_lst):],clim[:preds.shape[0], p, -len(targ_lst):]) for p in range(preds.shape[1])]

    print(len(rmses))
    print(len(dataset.ds_ecland['x']))
    
    performance_total = pd.DataFrame()
    performance_total['x'] = dataset.ds_ecland['x'] #dataset.x_idxs
    performance_total['lat'] = dataset.lats
    performance_total['lon'] = dataset.lons
    performance_total['rmse'] = rmses
    performance_total['mae'] = maes
    performance_total['r2'] = r2
    if clim is not None:
        performance_total['acc'] = acc
    if save_to is not None:
        performance_total.to_csv(os.path.join(save_to, f'scores_spatial_total.csv'))

    performance_targetwise = {}
    
    if targetwise:
        
        for targ in range(len(targ_lst)):
            
            print("calculating target scores:", targ_lst[targ])

            rmses = [mean_squared_error(preds[:,p, -(len(targ_lst))+targ], obs[:, p,-(len(targ_lst))+targ], squared=False) for p in range(obs.shape[1])]
            maes = [mean_absolute_error(preds[:,p, -(len(targ_lst))+targ], obs[:, p,-(len(targ_lst))+targ]) for p in range(obs.shape[1])]
            r2 = [r2_score(preds[:,p, -(len(targ_lst))+targ], obs[:, p,-(len(targ_lst))+targ]) for p in range(obs.shape[1])]
            if clim is not None:
                acc = [anomaly_correlation(preds[:, p, -len(targ_lst)+targ], obs[:, p, -len(targ_lst)+targ],clim[:preds.shape[0], p, -len(targ_lst)+targ]) for p in range(preds.shape[1])]

            performance = pd.DataFrame()
            performance = pd.DataFrame()
            performance['x'] = dataset.ds_ecland['x']# dataset.x_idxs
            performance['lat'] = dataset.lats
            performance['lon'] = dataset.lons
            performance['rmse'] = rmses
            performance['mae'] = maes
            performance['r2'] = r2
            if clim is not None:
                performance['acc'] = acc
            if save_to is not None:
                performance.to_csv(os.path.join(save_to, f'scores_spatial_{targ_lst[targ]}.csv'))

            performance_targetwise[targ_lst[targ]] = performance
    
    return performance_total, performance_targetwise


def get_scores_temporal(preds, obs, dataset, clim = None, save_to = ''):
    
    targ_lst = dataset.targ_lst

    if clim is not None:
        acc = [anomaly_correlation(preds[t, :, -len(targ_lst):], obs[t, :, -len(targ_lst):],clim[t, :, -len(targ_lst):]) for t in range(preds.shape[0])]
        
    rmses = [mean_squared_error(preds[t,:, -len(targ_lst):], obs[t, :,-len(targ_lst):], squared=False) for t in range(obs.shape[0])]
    maes = [mean_absolute_error(preds[t,:, -len(targ_lst):], obs[t, :,-len(targ_lst):]) for t in range(obs.shape[0])]
    
    performance_total = pd.DataFrame()
    performance_total['time'] = dataset.times[:preds.shape[0]]
    performance_total['rmse'] = rmses
    performance_total['mae'] = maes
    if clim is not None:
        performance_total['acc'] = acc
    if save_to is not None:
        performance_total.to_csv(os.path.join(save_to, f'scores_temporal_total.csv'))

    scores_total = {"rmse":rmses, "mae":maes, "acc":acc}
    
    print("Targetwise errors ")
    
    scores_targetwise = {}
    
    for targ in range(len(targ_lst)):

        if clim is not None:
            acc = [anomaly_correlation(preds[t, :, -len(targ_lst)+targ], obs[t, :, -len(targ_lst)+targ],clim[t, :, -len(targ_lst)+targ]) for t in range(preds.shape[0])]
        
        rmses = [mean_squared_error(preds[t,:, -len(targ_lst)+targ], obs[t, :,-len(targ_lst)+targ], squared=False) for t in range(obs.shape[0])]
        maes = [mean_absolute_error(preds[t,:, -len(targ_lst)+targ], obs[t, :,-len(targ_lst)+targ]) for t in range(obs.shape[0])]
          
        performance = pd.DataFrame()
        performance['time'] = dataset.times[:preds.shape[0]]
        performance['rmse'] = rmses
        performance['mae'] = maes
        if clim is not None:
            performance['acc'] = acc
        if save_to is not None:
            performance.to_csv(os.path.join(save_to, f'scores_temporal_{targ_lst[targ]}.csv'))

        scores = {"rmse":rmses, "mae":maes, "acc":acc}
        
        scores_targetwise[targ_lst[targ]] = scores
        
    return scores_total, scores_targetwise


def get_scores_temporal_cumulative(preds, obs, dataset, save_to = '', file= ''):
    
    targ_lst = dataset.targ_lst
    
    scores_targetwise = {}
    
    for targ in range(len(targ_lst)):
        
        print("Targetwise errors: ", targ_lst[targ])
        
        r2 = [r2_score_multi(preds[:t,:, -len(targ_lst)+targ], obs[:t, :,-len(targ_lst)+targ]) for t in range(1, obs.shape[0]+1)]
        rmses = [mean_squared_error(preds[:t,:, -len(targ_lst)+targ], obs[:t, :,-len(targ_lst)+targ], squared=False) for t in range(1,obs.shape[0]+1)]
        maes = [mean_absolute_error(preds[:t,:, -len(targ_lst)+targ], obs[:t, :,-len(targ_lst)+targ]) for t in range(1,obs.shape[0]+1)]
        mape = [mean_absolute_percentage_error(obs[:t, :,-len(targ_lst)+targ], preds[:t,:, -len(targ_lst)+targ]) for t in range(1,obs.shape[0]+1)]
          
        performance = pd.DataFrame()
        performance['time'] = dataset.full_times[:preds.shape[0]]
        performance['rmse'] = rmses
        performance['mae'] = maes
        performance['r2'] = r2
        performance['mape'] = mape
        
        if save_to is not None:
            performance.to_csv(os.path.join(save_to, f'scores_temporally_integrated_{targ_lst[targ]}_{file}.csv'))

        scores = {"rmse":rmses, "mae":maes, "mape":mape, "r2":r2}
        
        scores_targetwise[targ_lst[targ]] = scores
        
    return scores_targetwise


def assemble_scores(dict_of_scores_by_model):
    
    assembled_scores = pd.DataFrame()
    assembled_scores['model'] = dict_of_scores_by_model.keys()
    assembled_scores['RMSE'] = [dict_of_scores_by_model[key]['rmse'].mean() for key in dict_of_scores_by_model]
    assembled_scores['MAE'] = [dict_of_scores_by_model[key]['mae'].mean() for key in dict_of_scores_by_model]
    assembled_scores['R2'] = [dict_of_scores_by_model[key]['r2'].mean() for key in dict_of_scores_by_model]
    try:
        assembled_scores['ACC'] = [dict_of_scores_by_model[key]['acc'].mean() for key in dict_of_scores_by_model]
    except KeyError:
        pass
    assembled_scores = assembled_scores.set_index('model')
    
    assembled_scores.round(3)
    
    return assembled_scores


def assemble_scores_targetwise(dict_of_scores_by_model_targetwise, targ):
    
    assembled_scores = pd.DataFrame()
    assembled_scores['model'] = dict_of_scores_by_model_targetwise.keys()
    assembled_scores['RMSE'] = [dict_of_scores_by_model_targetwise[key][targ]['rmse'].mean() for key in dict_of_scores_by_model_targetwise]
    assembled_scores['MAE'] = [dict_of_scores_by_model_targetwise[key][targ]['mae'].mean() for key in dict_of_scores_by_model_targetwise]
    assembled_scores['R2'] = [dict_of_scores_by_model_targetwise[key][targ]['r2'].mean() for key in dict_of_scores_by_model_targetwise]
    try:
        assembled_scores['ACC'] = [dict_of_scores_by_model_targetwise[key][targ]['acc'].mean() for key in dict_of_scores_by_model_targetwise]
    except KeyError:
        pass
    assembled_scores = assembled_scores.set_index('model')
    
    assembled_scores.round(3)
    
    return assembled_scores


def anomaly(y_hat, climatology):
    
    return y_hat - climatology
    
def standardized_anomaly(y_hat, climatology, climatology_std):
    
    return anomaly(y_hat, climatology)/climatology_std

def anomaly_correlation(forecast, reference, climatology):
    
    anomaly_f = anomaly(forecast, climatology)
    anomaly_r = anomaly(reference, climatology)
    
    msse = np.mean(anomaly_f * anomaly_r)
    act = np.sqrt(np.mean(anomaly_f**2) * np.mean(anomaly_r**2))
    
    return msse/act


def rmse_saturation(reference, climatology):
    
    anomaly_r = anomaly(reference, climatology)
    A_r_squared =  np.mean(anomaly_r**2)
    A_r = np.sqrt(A_r_squared)
    saturation_level = A_r*np.sqrt(2)
    
    return saturation_level

