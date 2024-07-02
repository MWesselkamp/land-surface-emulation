"""
Script to create forecast horizon plots. Requires multiprocessing library.
For long lead times, run on a node with 256Gb memory.
"""

import torch.multiprocessing as mp

# Set the start method to 'spawn' for run on GPU partition.
mp.set_start_method('spawn', force=True)

import os
import time
import sys
import xarray as xr
import numpy as np
import torch
import yaml
import argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
print(SCRIPT_DIR)

from evaluation.evaluation_module import EvaluationModule
from evaluation.forecast_module import ForecastModule
from evaluation.helpers import load_model_from_checkpoint, load_model_with_config
from data.data_module import EcDataset, NonLinRegDataModule
from multiprocessing import Pool

parser = argparse.ArgumentParser(description="Load different config files for training.")
parser.add_argument('--config_file', type=str, help='Specify .yaml file from same directory.')
parser.add_argument('--target', type=str, help='Specify target to compute horizons on.')
parser.add_argument('--score', type=str, help='Specify score that will be computed.')
parser.add_argument('--num_cpus', type=int, help='Specify num cpus.')

args = parser.parse_args()

path_to_results = 'src/evaluation/analyses/europe/forecast_horizons' # None
configs_path = 'configs'

print("Computing forecast horizons for config:", args.config_file)
print("Computing forecast horizons for target:", args.target)
print("Computing forecast horizons with score:", args.score)
print("Computing forecast horizons with num_cpus:", args.num_cpus)
    

def init_worker(module):
    global evaluation_module
    evaluation_module = module

def process_input_total(i):
    result = evaluation_module.iterate_initial_time_total(i)
    return result

def process_input_target(i):
    result = evaluation_module.iterate_initial_time_target(i)
    return result
    

if __name__ == '__main__':

    with open(os.path.join(configs_path, args.config_file)) as stream:
        try:
            CONFIG = yaml.safe_load(stream)
            print(f"Opening {args.config_file} for experiment configuration.")
        except yaml.YAMLError as exc:
            print(exc)

    dataset = EcDataset(CONFIG, 
                       CONFIG['test_start'],
                        CONFIG['test_end'])

    model = load_model_with_config(CONFIG)
    forecast_module = ForecastModule(model)

    time_idxs = 1350  #(dataset.time_size)
    print("Time idxs:", time_idxs ) # 1350

    # specify evaluation module outside of process_input function.
    module = EvaluationModule(forecast_module, score = args.score)
    module.set_test_data(dataset)
    module.set_climatology(CONFIG['climatology_path'])
    module.set_lead_time(time_idxs)
    
    num_iterations = time_idxs  # total number of iterations
    num_cpus = args.num_cpus  # os.cpu_count()
    inputs = range(num_iterations)

    start_time = time.time()
    print("Start horizons computation.")
    
    if args.target == 'total':
        with Pool(num_cpus, initializer=init_worker, initargs=(module,)) as pool:
            scores_total = pool.map(process_input_total, inputs)
            
    elif args.target in CONFIG['targets_prog']:
        module.set_target(args.target, CONFIG['targets_prog'])
        with Pool(num_cpus, initializer=init_worker, initargs=(module,)) as pool:
            scores_total = pool.map(process_input_target, inputs)

    else:
        print("Don't know prognostic variable!")

    end_time = time.time()
    duration = (end_time - start_time)/60
    print("Time required for horizons compuation ... minutes ...:", duration)

    times = np.array(dataset.times[CONFIG['lookback']:time_idxs])

    print(times)
    module.plot_heatmap(scores = scores_total,
                        times = times,
                        discrete_classes = 10,
                        threshold = 1,
                        style = 'rectangle', 
                        cmap_style = 'OrRd_r',
                        save_to = path_to_results, 
                        filename = f"{CONFIG['model']}_{CONFIG['logging']['region']}_{args.target}")

    module.plot_timeseries(save_to = os.path.join(path_to_results, f"{CONFIG['model']}_{CONFIG['logging']['region']}_timeseries.pdf"))


 
   



    
    