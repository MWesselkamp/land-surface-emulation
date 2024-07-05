import os
import sys
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cartopy.crs as ccrs
import xgboost as xgb
import torch
import yaml
import argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
print(SCRIPT_DIR)

from evaluation.forecast_module import ForecastModule
from evaluation.helpers import load_model_from_checkpoint, load_model_with_config
from data.data_module import EcDataset, NonLinRegDataModule
from utils.visualise import make_ailand_plot,make_ailand_plot_combined, plot_score_map, plot_losses, plot_losses_targetwise,plot_losses_targetwise_boxplots, boxplot_scores_single, boxplot_scores_single_new, plot_scores_temporal_targetwise_cumulative, plot_correlation_scatter, plot_correlation_scatter_binned
from utils.utils import r2_score_multi, get_scores_spatial_global, get_scores_temporal, assemble_scores

def nullable_string(val):
    if not val:
        return None
    return val

parser = argparse.ArgumentParser(description="Load different config files for training.")

parser.add_argument('--config_file_mlp', type=nullable_string, help='Specify .yaml file from same directory.')
parser.add_argument('--config_file_lstm', type=nullable_string, help='Specify .yaml file from same directory.')
parser.add_argument('--config_file_xgb', type=nullable_string, help='Specify .yaml file from same directory.')

args = parser.parse_args()

if args.config_file_mlp == 'None':
    args.config_file_mlp = None
if args.config_file_lstm == 'None':
    args.config_file_lstm = None
if args.config_file_xgb == 'None':
    args.config_file_xgb = None
    
configs_path = 'src/configs'
path_to_figures = 'src/evaluation/figures'

print('config_file_mlp', args.config_file_mlp)
print('config_file_lstm', args.config_file_lstm)
print('config_file_xgb', args.config_file_xgb)

if args.config_file_mlp is not None:

    with open(os.path.join(configs_path, args.config_file_mlp)) as stream:
        try:
            CONFIG = yaml.safe_load(stream)
            print(f"Opening {args.config_file_mlp} for experiment configuration.")
        except yaml.YAMLError as exc:
            print(exc)

    data_module = NonLinRegDataModule(CONFIG)
    dataset = EcDataset(CONFIG, 
                       CONFIG['test_start'],
                       CONFIG['test_end'])

    losses = pd.read_csv(os.path.join(CONFIG['model_path'], 'metrics.csv'))

    plot_losses(losses, save_to = CONFIG['model_path'])
    plot_losses_targetwise(losses, save_to = CONFIG['model_path'], label = "SmoothL1")
    plot_losses_targetwise_boxplots(losses, save_to = CONFIG['model_path'], label = "SmoothL1", log = False)
    
    mlp_model = load_model_with_config(CONFIG)
    mlp_module = ForecastModule(mlp_model)
    # restructure this if possible. Why return the data here and not just call get_test_data when creating the module with model and dataset?
    X_static, X_met, Y_prog = mlp_module.get_test_data(dataset)
    mlp_prog, mlp_prog_prediction = mlp_module.step_forecast(X_static, X_met, Y_prog)

    make_ailand_plot(mlp_prog_prediction[:, 85:95, :], 
                     mlp_prog[:, 85:95, :], 
                     mlp_prog.shape[-1],
                      save_to = os.path.join(CONFIG['model_path'], 'ailand_plot.pdf'))

    climatology = mlp_module.get_climatology(CONFIG['climatology_path'])
    print(climatology)
    # Make the computation of scores part of the evaluation module.
    mlp_performance_total, mlp_performance_targetwise = get_scores_spatial_global(mlp_prog_prediction, mlp_prog, dataset, climatology,
                                                                   targetwise = True, save_to = CONFIG['model_path'])

    mlp_performance_total_temp, mlp_performance_targetwise_temp = get_scores_temporal(mlp_prog_prediction, mlp_prog,
                                                                                    dataset, climatology, save_to = CONFIG['model_path'])

    scores_by_model = {'MLP': mlp_performance_total}
    assembled_scores = assemble_scores(scores_by_model)
    print(assembled_scores)
    print("")

    for targ in CONFIG['targets_prog']:
        print(targ)        
        targ_scores_by_model = {'MLP': mlp_performance_targetwise[targ]}
        assembled_scores = assemble_scores(targ_scores_by_model)
        print(assembled_scores)
        print("")

    plot_score_map(mlp_performance_total, error='acc', vmin = 0, vmax = None, cmap = "PuOr", transparent = True, save_to = CONFIG['model_path'], file=f'total', ax=None, cb = 'cb')
    plot_score_map(mlp_performance_total, error='r2', vmin = 0.98,vmax = None, cmap = "PuOr", transparent = True, save_to = CONFIG['model_path'], file=f'total', ax=None, cb = 'cb')
    plot_score_map(mlp_performance_total, error='rmse', vmin = 0,vmax = None, cmap = "PuOr_r", transparent = True, save_to = CONFIG['model_path'], file=f'total', ax=None, cb = 'cb')
    plot_score_map(mlp_performance_total, error='mae', vmin = 0,vmax = None, cmap = "PuOr_r", transparent = True, save_to = CONFIG['model_path'], file=f'total', ax=None, cb = 'cb')

if args.config_file_lstm is not None:
    

    with open(os.path.join(configs_path, args.config_file_lstm)) as stream:
        try:
            CONFIG = yaml.safe_load(stream)
            print(f"Opening {args.config_file_lstm} for experiment configuration.")
        except yaml.YAMLError as exc:
            print(exc)
    
    #data_module = NonLinRegDataModule(CONFIG)
    dataset = EcDataset(CONFIG, 
                   CONFIG['test_start'],
                   CONFIG['test_end'])

    losses = pd.read_csv(os.path.join(CONFIG['model_path'], 'metrics.csv'))

    plot_losses(losses, which='loss_logit', save_to = CONFIG['model_path'])
    plot_losses_targetwise(losses, save_to = CONFIG['model_path'], label = "SmoothL1")
    plot_losses_targetwise_boxplots(losses, save_to = CONFIG['model_path'], label = "SmoothL1", log = False)
    
    lstm_model = load_model_with_config(CONFIG, my_device = 'cpu')
    lstm_module = ForecastModule(lstm_model, my_device = 'cpu')

    # Clean this!
    if CONFIG['logging']['name'] in ('global_highres', 'global', 'global_1h'):

        total_size = dataset.x_size
        print("TOTAL DATA SIZE:", total_size)
        chunk_size = 23000  
    
        # Calculate the number of chunks needed to cover the full data set
        num_chunks = (total_size + chunk_size - 1) // chunk_size 
        print("SPLIT DATA IN CHUNKS:", num_chunks)
        
        lstm_prog_chunks = []
        lstm_prog_prediction_chunks = []
    
        for i in range(num_chunks):
            
            # Calculate the start and end indices for each chunk
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, total_size)
            print("CHUNK start:", start_idx)
            print("CHUNK end:", end_idx)
    
            # Update the configuration with the current slice indices
            config_temp = CONFIG.copy()
            config_temp["x_slice_indices"] = (start_idx, end_idx)
    
            dataset_temp = EcDataset(config_temp, 
                                     config_temp['test_start'],
                                     config_temp['test_end'])
    
            # Get the test data for the current chunk
            X_static, X_met, Y_prog = lstm_module.get_test_data(dataset_temp, chunk_idx = (start_idx, end_idx))

            lstm_prog_chunk, lstm_prog_prediction_chunk = lstm_module.step_forecast(X_static, X_met, Y_prog)
    
            lstm_prog_chunks.append(lstm_prog_chunk)
            lstm_prog_prediction_chunks.append(lstm_prog_prediction_chunk)
    
        lstm_prog = np.concatenate(lstm_prog_chunks, axis=1)
        lstm_prog_prediction = np.concatenate(lstm_prog_prediction_chunks, axis=1)        
    
    elif (CONFIG['logging']['name'] ==  'europe'):
        X_static, X_met, Y_prog = lstm_module.get_test_data(dataset)
        print("X_static:", X_static.shape)
        print("X_met:", X_met.shape)
        print("Y_prog:", Y_prog.shape)
        lstm_prog, lstm_prog_prediction = lstm_module.step_forecast(X_static, X_met, Y_prog)
    else:
        print("DONT KNOW HOW TO LOAD DATA")

    make_ailand_plot(lstm_prog_prediction[:, 85:95, :], 
                     lstm_prog[:, 85:95, :], 
                     lstm_prog.shape[-1],
                      save_to = os.path.join(CONFIG['model_path'], 'ailand_plot.pdf'))

    climatology = lstm_module.get_climatology(CONFIG['climatology_path'])
    lstm_performance_total, lstm_performance_targetwise = get_scores_spatial_global(lstm_prog_prediction, lstm_prog,
                                                                                    dataset, climatology,
                                                                   targetwise = True, save_to = CONFIG['model_path'])

    lstm_performance_total_temp, lstm_performance_targetwise_temp = get_scores_temporal(lstm_prog_prediction, lstm_prog,
                                                                                    dataset, climatology, save_to = CONFIG['model_path'])
    
    scores_by_model = {'LSTM': lstm_performance_total}
    assembled_scores = assemble_scores(scores_by_model)
    print(assembled_scores)
    print("")

    for targ in CONFIG['targets_prog']:
        print(targ)
        targ_scores_by_model = {'LSTM': lstm_performance_targetwise[targ]}
        assembled_scores = assemble_scores(targ_scores_by_model)
        print(assembled_scores)
        print("")

    lstm_best_gridcell =  np.argmin(np.array(lstm_performance_total['rmse']))
    lstm_worst_gridcell =  np.argmax(np.array(lstm_performance_total['rmse']))
    print("Worst grid cell in RMSE:", np.argmax(np.array(lstm_performance_total['rmse'])))
    print("Best grid cell in RMSE:", np.argmin(np.array(lstm_performance_total['rmse'])))
    print("Worst grid cell in R2:", np.argmin(np.array(lstm_performance_total['r2'])))
    print("Best grid cell in R2:", np.argmax(np.array(lstm_performance_total['r2'])))

    if climatology is not None:
        plot_score_map(lstm_performance_total, error='acc', vmin = 0, vmax = None, cmap = "PuOr", transparent = True, save_to = CONFIG['model_path'], file=f'total', ax=None, cb = 'cb')
    plot_score_map(lstm_performance_total, error='r2', vmin = 0.98,vmax = None, cmap = "PuOr", transparent = True, save_to = CONFIG['model_path'], file=f'total', ax=None, cb = 'cb')
    plot_score_map(lstm_performance_total, error='rmse', vmin = 0,vmax = None, cmap = "PuOr_r", transparent = True, save_to = CONFIG['model_path'], file=f'total', ax=None, cb = 'cb')
    plot_score_map(lstm_performance_total, error='mae', vmin = 0,vmax = None, cmap = "PuOr_r", transparent = True, save_to = CONFIG['model_path'], file=f'total', ax=None, cb = 'cb')

if args.config_file_xgb is not None:

    with open(os.path.join(configs_path, args.config_file_xgb)) as stream:
        try:
            CONFIG = yaml.safe_load(stream)
            print(f"Opening {args.config_file_xgb} for experiment configuration.")
        except yaml.YAMLError as exc:
            print(exc)

    data_module = NonLinRegDataModule(CONFIG)
    dataset = EcDataset(CONFIG, 
                       CONFIG['test_start'],
                       CONFIG['test_end'])

    xgb_model = load_model_with_config(CONFIG, my_device = 'cpu')

    xgb_module = ForecastModule(xgb_model, my_device = 'cpu')
    X_static, X_met, Y_prog = xgb_module.get_test_data(dataset)
    print("X_static:", X_static.shape)
    print("X_met:", X_met.shape)
    print("Y_prog:", Y_prog.shape)
    xgb_prog, xgb_prog_prediction = xgb_module.step_forecast(X_static, X_met, Y_prog)

    make_ailand_plot(xgb_prog_prediction[:, 85:95, :], 
                     xgb_prog[:, 85:95, :], 
                     xgb_prog.shape[-1],
                      save_to = os.path.join(CONFIG['model_path'], 'ailand_plot.pdf'))

    climatology = xgb_module.get_climatology(CONFIG['climatology_path'])
    xgb_performance_total, xgb_performance_targetwise = get_scores_spatial_global(xgb_prog_prediction, xgb_prog, dataset, climatology,
                                                                   targetwise = True, save_to = CONFIG['model_path'])

    xgb_performance_total_temp, xgb_performance_targetwise_temp = get_scores_temporal(xgb_prog_prediction, xgb_prog,
                                                                                    dataset, climatology, save_to = CONFIG['model_path'])
    
    scores_by_model = {'XGB': xgb_performance_total}
    assembled_scores = assemble_scores(scores_by_model)
    print(assembled_scores)
    print("")

    for targ in CONFIG['targets_prog']:
        print(targ)
        targ_scores_by_model = {'XGB': xgb_performance_targetwise[targ]}
        assembled_scores = assemble_scores(targ_scores_by_model)
        print(assembled_scores)
        print("")

    print("Worst grid cell in RMSE:", np.argmax(np.array(xgb_performance_total['rmse'])))
    print("Best grid cell in RMSE:", np.argmin(np.array(xgb_performance_total['rmse'])))
    print("Worst grid cell in R2:", np.argmin(np.array(xgb_performance_total['r2'])))
    print("Best grid cell in R2:", np.argmax(np.array(xgb_performance_total['r2'])))

    plot_score_map(xgb_performance_total, error='acc', vmin = 0,vmax = None, cmap = "PuOr", transparent = True, save_to = CONFIG['model_path'], file=f'total', ax=None, cb = 'cb')
    plot_score_map(xgb_performance_total, error='r2', vmin = 0.98,vmax = None, cmap = "PuOr", transparent = True, save_to = CONFIG['model_path'], file=f'total', ax=None, cb = 'cb')
    plot_score_map(xgb_performance_total, error='rmse', vmin = 0,vmax = None, cmap = "PuOr_r", transparent = True, save_to = CONFIG['model_path'], file=f'total', ax=None, cb = 'cb')
    plot_score_map(xgb_performance_total, error='mae', vmin = 0,vmax = None, cmap = "PuOr_r", transparent = True, save_to = CONFIG['model_path'], file=f'total', ax=None, cb = 'cb')


if any(config is None for config in [args.config_file_mlp, args.config_file_lstm, args.config_file_xgb]):
    print("Missing a config file for assembling results.")
    print("")
    print("Finish run.")
else:
    print("Assembling results.")
    print("")
    print("Plot timeseries.")
    make_ailand_plot_combined(lstm_prog[:, lstm_worst_gridcell, np.newaxis, :], 
                              xgb_prog_prediction[:, lstm_worst_gridcell, np.newaxis,  :],
                              mlp_prog_prediction[:, lstm_worst_gridcell, np.newaxis, :],
                              lstm_prog_prediction[:, lstm_worst_gridcell,np.newaxis,  :], 
                              target_size = 7,
                              period = dataset.times,
                              save_to = os.path.join(path_to_figures, 'europe'),
                             filename = 'ailand_plot_combined_worstgridcell.pdf')

    make_ailand_plot_combined(lstm_prog[:, lstm_best_gridcell, np.newaxis, :], 
                              xgb_prog_prediction[:, lstm_best_gridcell, np.newaxis,  :],
                              mlp_prog_prediction[:, lstm_best_gridcell, np.newaxis, :],
                              lstm_prog_prediction[:, lstm_best_gridcell,np.newaxis,  :], 
                              target_size = 7,
                              period = dataset.times,
                              save_to = os.path.join(path_to_figures, 'europe'),
                             filename = 'ailand_plot_combined_bestgridcell.pdf')

    print("")
    print("MEAN SCORES")
    print("")
    scores_by_model = {'XGB':xgb_performance_total, 
                       'MLP': mlp_performance_total, 
                       'LSTM': lstm_performance_total, 
                      }
    assembled_scores = assemble_scores(scores_by_model)
    print(assembled_scores.round(4))
    print("")
    
    for targ in CONFIG['targets_prog']:
        print(targ)
        targ_scores_by_model = {'XGB':xgb_performance_targetwise[targ], 
                                'MLP': mlp_performance_targetwise[targ], 
                                'LSTM': lstm_performance_targetwise[targ]}
        assembled_scores = assemble_scores(targ_scores_by_model)
        print(assembled_scores.round(4))
        print("")
    
    fig, axs = plt.subplots(1, 3, figsize=(24, 8))
    plot_scores_temporal_targetwise_cumulative(xgb_performance_targetwise_temp, score = 'acc', ax = axs[0], file = 'XGB')
    plot_scores_temporal_targetwise_cumulative(mlp_performance_targetwise_temp, score = 'acc', ax = axs[1], file = 'MLP')
    plot_scores_temporal_targetwise_cumulative(lstm_performance_targetwise_temp, score = 'acc', ax = axs[2], file = 'LSTM')
    plt.tight_layout()
    plt.savefig(os.path.join(path_to_figures, f"scores_temporal_combined_acc.pdf"))
    plt.show()
    plt.close()

    scores = ['r2', 'acc', 'rmse', 'mae']
    
    for s in scores:
        
        vmin = 0.98 if s == 'r2' else 0        
        fig, axs = plt.subplots(1, 3, figsize=(24, 8), subplot_kw={'projection': ccrs.PlateCarree()})
        plot_score_map(xgb_performance_total, error=s, vmin = vmin,vmax = None, cmap = "PuOr", transparent = True, 
                   save_to = path_to_figures, file=f'total_xgb', ax= axs[0])
        plot_score_map(mlp_performance_total, error=s, vmin = vmin,vmax = None, cmap = "PuOr", transparent = True, 
                   save_to = path_to_figures, file=f'total_mlp', ax= axs[1])
        plot_score_map(lstm_performance_total, error=s, vmin = vmin,vmax = None, cmap = "PuOr", transparent = True, 
                   save_to = path_to_figures, file=f'total_lstm', ax= axs[2])
        plt.tight_layout()
        plt.savefig(os.path.join(path_to_figures, f"score_maps_combined_{s}.pdf"))
        plt.show()
        plt.close()

        for targ in CONFIG['targets_prog']:

            fig, axs = plt.subplots(1, 3, figsize=(24, 8), subplot_kw={'projection': ccrs.PlateCarree()})
            plot_score_map(xgb_performance_targetwise[targ], error=s, vmin = vmin,vmax = None, cmap = "PuOr", transparent = True, 
                   save_to = path_to_figures, file=f'total_xgb', ax= axs[0])
            plot_score_map(mlp_performance_targetwise[targ], error=s, vmin = vmin,vmax = None, cmap = "PuOr", transparent = True, 
                   save_to = path_to_figures, file=f'total_mlp', ax= axs[1])
            plot_score_map(lstm_performance_targetwise[targ], error=s, vmin = vmin,vmax = None, cmap = "PuOr", transparent = True, 
                   save_to = path_to_figures, file=f'total_lstm', ax= axs[2])
            plt.tight_layout()
            plt.savefig(os.path.join(path_to_figures, f"score_maps_combined_{s}_{targ}.pdf"))
            plt.show()
            plt.close()
            
        boxplot_scores_single_new(xgb_performance_total, 
                             mlp_performance_total, 
                             lstm_performance_total, 
                             score = s,
                             log= False if s in ['r2', 'acc'] else True, 
                              data='europe', target = 'total', save_to = path_to_figures)

        

        i = 0
        targ_lst_labels = ["Soil Moisture Layer 1", "Soil Moisture Layer 2", "Soil Moisture Layer 3", 
                       "Soil Temperature Layer 1", "Soil Temperature Layer 2", "Soil Temperature Layer 3", 
                       "Snow Cover Fraction"]
        for targ in CONFIG['targets_prog']:
            boxplot_scores_single_new(xgb_performance_targetwise[targ], 
                                 mlp_performance_targetwise[targ], 
                                 lstm_performance_targetwise[targ], 
                                 score = s,
                                 log= False if s in ['r2', 'acc'] else True, 
                                  data='europe', target = targ, save_to = path_to_figures)

            plot_correlation_scatter_binned([xgb_prog[:, :, i], mlp_prog[:, :, i], lstm_prog[:, :, i]], 
                                        [xgb_prog_prediction[:, :, i], mlp_prog_prediction[:, :, i], lstm_prog_prediction[:, :, i]], 
                                        targ_lst_labels[i], save_to = os.path.join(path_to_figures, f"correlations_{targ}.pdf"))

            i += 1




