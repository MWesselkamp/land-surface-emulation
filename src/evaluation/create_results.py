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

from evaluation.forecast_module import *
from evaluation.helpers import *
from data.data_module import *
from utils.visualise import *
from utils.utils import *

# Helper function for nullable string
def nullable_string(val):
    return None if not val else val

# Argument parser setup
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
    
configs_path = 'configs'
path_to_figures = 'src/evaluation/figures/europe'

print('config_file_mlp', args.config_file_mlp)
print('config_file_lstm', args.config_file_lstm)
print('config_file_xgb', args.config_file_xgb)

if args.config_file_mlp is not None:

    CONFIG = load_config(configs_path, args.config_file_mlp)

    losses = pd.read_csv(os.path.join(CONFIG['model_path'], 'metrics.csv'))
    plot_losses_and_metrics(losses, config= CONFIG)
    
    mlp_performance_total, mlp_performance_targetwise, mlp_prog, mlp_prog_prediction = load_and_process_model(args.config_file_mlp, configs_path, model_name= CONFIG['model'])

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
    
    CONFIG = load_config(configs_path, args.config_file_lstm)

    losses = pd.read_csv(os.path.join(CONFIG['model_path'], 'metrics.csv'))
    plot_losses_and_metrics(losses, config= CONFIG)
        
    lstm_performance_total, lstm_performance_targetwise, lstm_prog, lstm_prog_prediction = load_and_process_model(args.config_file_lstm, configs_path, model_name= CONFIG['model'])
    
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
    print_best_and_worst_gridcells(lstm_performance_total)

    plot_score_map(lstm_performance_total, error='acc', vmin = 0, vmax = None, cmap = "PuOr", transparent = True, save_to = CONFIG['model_path'], file=f'total', ax=None, cb = 'cb')
    plot_score_map(lstm_performance_total, error='r2', vmin = 0.98,vmax = None, cmap = "PuOr", transparent = True, save_to = CONFIG['model_path'], file=f'total', ax=None, cb = 'cb')
    plot_score_map(lstm_performance_total, error='rmse', vmin = 0,vmax = None, cmap = "PuOr_r", transparent = True, save_to = CONFIG['model_path'], file=f'total', ax=None, cb = 'cb')
    plot_score_map(lstm_performance_total, error='mae', vmin = 0,vmax = None, cmap = "PuOr_r", transparent = True, save_to = CONFIG['model_path'], file=f'total', ax=None, cb = 'cb')

if args.config_file_xgb is not None:

    CONFIG = load_config(configs_path, args.config_file_xgb)
    dataset = EcDataset(CONFIG, CONFIG['test_start'], CONFIG['test_end'])

    xgb_performance_total, xgb_performance_targetwise, xgb_prog, xgb_prog_prediction = load_and_process_model(args.config_file_xgb, configs_path, model_name=  CONFIG['model'])
    
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

    print_best_and_worst_gridcells(xgb_performance_total)

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
                              save_to = path_to_figures,
                              filename = 'ailand_plot_combined_worstgridcell.pdf')

    make_ailand_plot_combined(lstm_prog[:, lstm_best_gridcell, np.newaxis, :], 
                              xgb_prog_prediction[:, lstm_best_gridcell, np.newaxis,  :],
                              mlp_prog_prediction[:, lstm_best_gridcell, np.newaxis, :],
                              lstm_prog_prediction[:, lstm_best_gridcell,np.newaxis,  :], 
                              target_size = 7,
                              period = dataset.times,
                              save_to = path_to_figures,
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

    # Aggregate list of total and targetwise model scores
    model_scores_total = [xgb_performance_total, mlp_performance_total, lstm_performance_total]
    model_scores_targetwise = [xgb_performance_targetwise, mlp_performance_targetwise, lstm_performance_targetwise]
    
    # Target labels for plotting
    targ_lst_labels = [
        "Soil Moisture Layer 1", "Soil Moisture Layer 2", "Soil Moisture Layer 3", 
        "Soil Temperature Layer 1", "Soil Temperature Layer 2", "Soil Temperature Layer 3", 
        "Snow Cover Fraction"
    ]

    scores = ['r2', 'acc', 'rmse', 'mae']
    
    for score in scores:
        
        vmin = 0.98 if score == 'r2' else 0      

        plot_model_map_comparison(model_scores_total, score, vmin, path_to_figures, filename_prefix = score, target='total')

        for target in CONFIG['targets_prog']:

            models_performance_target = [model[target] for model in model_scores_targetwise]

            plot_model_map_comparison(models_performance_target, score, vmin, path_to_figures, filename_prefix = score, target=target)

            boxplot_scores_single_new(xgb_performance_targetwise[target], mlp_performance_targetwise[target], lstm_performance_targetwise[target],
                score=score, log=False if score in ['r2', 'acc'] else True, data='europe', target=target, save_to=path_to_figures
            )

            
        boxplot_scores_single_new(xgb_performance_total, mlp_performance_total, lstm_performance_total, 
            score=score, log=False if score in ['r2', 'acc'] else True, data='europe', target='total', save_to=path_to_figures
    )

        # Scatter plot correlations for each target
        for i, target in enumerate(CONFIG['targets_prog']):
            plot_correlation_scatter_binned(
                [xgb_prog[:, :, i], mlp_prog[:, :, i], lstm_prog[:, :, i]],
                [xgb_prog_prediction[:, :, i], mlp_prog_prediction[:, :, i], lstm_prog_prediction[:, :, i]],
                targ_lst_labels[i], save_to=os.path.join(path_to_figures, f"correlations_{target}.pdf")
            )




