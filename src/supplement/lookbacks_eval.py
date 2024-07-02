import pandas as pd
import numpy as np
import xarray as xr
import os
import sys
import yaml
import torch
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from torch import tensor
from evaluation.forecast_module import ForecastModule
from evaluation.helpers import load_model_from_checkpoint, load_model_with_config
from lstm_2D.models import LSTM_m2m_global as LSTMregressor
from data.data_module import EcDataset, NonLinRegDataModule
from utils.visualise import boxplot_scores_leadtimes, plot_losses_combined

from utils.utils import r2_score_multi, get_scores_spatial
from utils.utils import assemble_scores, assemble_scores_targetwise

if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
        
db_path = "/perm/daep/ec_land_db_test/ecland_i6aj_2016_2022_europe.zarr" 

path_to_results = 'src/evaluation/analyses/europe/lookback/lookback_' # '../lstm_2D/results_m2m/europe/version_15'
configs_path = "src/configs"
config = 'lstm_europe_lookback_config.yaml'

targ_lst = ['swvl1',
            'swvl2',
            'swvl3',
            'stl1',
            'stl2',
            'stl3',
            'snowc'
            ]

lead_10 = os.path.join(os.path.join(path_to_results, '10/version_0'), 'metrics.csv')
lead_20 = os.path.join(os.path.join(path_to_results, '20/version_0'), 'metrics.csv')
lead_30 = os.path.join(os.path.join(path_to_results, '30/version_0'), 'metrics.csv')
lead_40 = os.path.join(os.path.join(path_to_results, '40/version_0'), 'metrics.csv')
lead_50 = os.path.join(os.path.join(path_to_results, '50/version_0'), 'metrics.csv')
lead_60 = os.path.join(os.path.join(path_to_results, '60/version_0'), 'metrics.csv')

training_leads = ["10", "20", "30", "40", "50", "60"]

plot_losses_combined([lead_10, lead_20, lead_30, lead_40, lead_50, lead_60], 
                     training_leads, 
                     which = 'loss_logit', label = "SmoothL1",save_to = path_to_results) 

with open(os.path.join(configs_path, config)) as stream:
        try:
            CONFIG = yaml.safe_load(stream)
            print(f"Opening {config} for experiment configuration.")
        except yaml.YAMLError as exc:
            print(exc)

device = torch.device(dev) if CONFIG["device"] is None else CONFIG["device"]

data_module = NonLinRegDataModule(CONFIG)
dataset = EcDataset(CONFIG, 
                    CONFIG['test_start'],
                    CONFIG['test_end'])


scores_by_model = {}
targ_scores_by_model = {}
lookback = [10, 20, 30, 40, 50, 60]
for l in range(len(training_leads)):

    CONFIG_TEMP = CONFIG
    CONFIG_TEMP["model_path"] = os.path.join(os.path.join(path_to_results, training_leads[l]), 'version_0')
    CONFIG_TEMP["lookback"] = lookback[l]
    model = load_model_with_config(CONFIG_TEMP)
    forecast_module = ForecastModule(model, my_device = 'cpu')
    X_static, X_met, Y_prog = forecast_module.get_test_data_global(dataset)
    Y_prog, Y_prog_prediction = forecast_module.step_forecast_global(X_static, X_met, Y_prog)
    
    #leadtime = load_model_from_checkpoint(os.path.join(path_to_results, training_leads[l]), modelname=training_leads[l])
    #leadtime_forecast = ForecastModule(leadtime)
    #static_features, dynamic_features = leadtime_forecast.get_test_data_europe(dataset)
    #dynamic_features, dynamic_features_prediction = leadtime_forecast.step_forecast(static_features, dynamic_features, time_idx_0 = 0, targ_lst = targ_lst)
    performance_total, performance_targetwise = get_scores_spatial(Y_prog_prediction,
                                                                   Y_prog, 
                                                                   dataset, 
                                                                   targetwise = True, 
                                                                   save_to = path_to_results)
    
    scores_by_model[training_leads[l]] = performance_total
    
    targ_scores_by_model[training_leads[l]] = performance_targetwise

    del Y_prog, Y_prog_prediction, X_static, X_met
    torch.cuda.empty_cache()

    
print("MEAN SCORES")
assembled_scores_total = assemble_scores(scores_by_model)
print(assembled_scores_total.round(4))

print("")
for targ in targ_lst:
    print("TARGET", targ)
    assembled_scores = assemble_scores_targetwise(targ_scores_by_model, targ)
    print(assembled_scores.round(4))
    print("")
    
boxplot_scores_leadtimes(scores_by_model, log = True, data='europe', 
                         target = None, save_to = path_to_results)
for targ in targ_lst:
    boxplot_scores_leadtimes(targ_scores_by_model, log = True, data='europe', 
                             target = targ, save_to = path_to_results)