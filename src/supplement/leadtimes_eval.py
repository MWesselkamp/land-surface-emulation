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
from forecast.forecast_module import ForecastModule
from forecast.helpers import load_model_from_checkpoint
from lstm_2D.models import LSTM_m2m_autoencoder_nc as LSTMregressor
from lstm_2D.ec_database import EcDataset, NonLinRegDataModule
from utils.visualise import boxplot_scores_leadtimes, plot_losses_combined

from utils.utils import r2_score_multi, get_scores_spatial
from utils.utils import assemble_scores, assemble_scores_targetwise

db_path = "/perm/daep/ec_land_db_test/ecland_i6aj_2016_2022_europe.zarr" 

path_to_results = 'src/evaluation/analyses/europe/leadtime' # '../lstm_2D/results_m2m/europe/version_15'


targ_lst = ['swvl1',
            'swvl2',
            'swvl3',
            'stl1',
            'stl2',
            'stl3',
            'snowc'
            ]

lead_10 = os.path.join(os.path.join(path_to_results, 'leadtime_10'), 'metrics.csv')
lead_20 = os.path.join(os.path.join(path_to_results, 'leadtime_20'), 'metrics.csv')
lead_30 = os.path.join(os.path.join(path_to_results, 'leadtime_30'), 'metrics.csv')
lead_40 = os.path.join(os.path.join(path_to_results, 'leadtime_40'), 'metrics.csv')
lead_50 = os.path.join(os.path.join(path_to_results, 'leadtime_50'), 'metrics.csv')
lead_60 = os.path.join(os.path.join(path_to_results, 'leadtime_60'), 'metrics.csv')

training_leads = ["leadtime_10", "leadtime_20", "leadtime_30", "leadtime_40", "leadtime_50", "leadtime_60"]

plot_losses_combined([lead_10, lead_20, lead_30, lead_40, lead_50, lead_60], 
                     training_leads, 
                     which = 'loss_logit', label = "SmoothL1",save_to = path_to_results) 

dataset = EcDataset(start_yr = "2022",
                   end_yr= "2022",
                    x_idxs=(0, None),
                    lat_lon = None,
                   path=db_path) 


scores_by_model = {}
targ_scores_by_model = {}

for l in range(len(training_leads)):
    leadtime = load_model_from_checkpoint(os.path.join(path_to_results, training_leads[l]), modelname=training_leads[l])
    leadtime_forecast = ForecastModule(leadtime)
    static_features, dynamic_features = leadtime_forecast.get_test_data_europe(dataset)
    dynamic_features, dynamic_features_prediction = leadtime_forecast.step_forecast(static_features, dynamic_features, time_idx_0 = 0, targ_lst = targ_lst)
    performance_total, performance_targetwise = get_scores_spatial(dynamic_features_prediction[12:,...], dynamic_features[12:,...], dataset, targetwise = True, save_to = path_to_results)
    
    scores_by_model[training_leads[l]] = performance_total
    
    targ_scores_by_model[training_leads[l]] = performance_targetwise
    
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