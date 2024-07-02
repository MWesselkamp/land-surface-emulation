import os
import sys
import time 

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
print(SCRIPT_DIR)

import xarray as xr
import pandas as pd
import xgboost as xgb
import numpy as np
import argparse
import yaml

from data.data_module import EcDataset
from baseline_2D.helpers import plot_feature_importances, plot_increments
from utils.utils import r2_score_multi

from sklearn.metrics import r2_score, mean_absolute_percentage_error, mean_absolute_error, mean_squared_error

parser = argparse.ArgumentParser(description="Load different config files for training.")
parser.add_argument('config_file', type=str, help='Specify .yaml file from same directory.')

args = parser.parse_args()

experiment_path = 'src/xgb'
configs_path = 'configs'

with open(os.path.join(configs_path, args.config_file)) as stream:
    try:
        CONFIG = yaml.safe_load(stream)
        print(f"Opening {args.config_file} for experiment configuration.")
    except yaml.YAMLError as exc:
        print(exc)
        
# db_path = "/perm/daep/ec_land_db_test/ecland_i6aj_2016_2022_europe.zarr" # "/perm/daep/ec_land_db_test/ecland_i6aj_2018_2022_6H.zarr"
path_to_results = os.path.join('src/xgb/results/global', CONFIG['logging']['name'])

def train_xgb(training_dataset, validation_dataset, path_to_results):
        
    # we need the data flattend for training, i.e. spatial is merged in temporal dimension.
    X_static, X, Y_prog, Y_inc = training_dataset.load_data()

    print("Statics shape:", X_static.shape)
    print("Statics shape:", X.shape)
    print("Statics shape:", Y_prog.shape)
    print("Statics shape:", Y_inc.shape)
    print("")
    print("Len dataset:", training_dataset.len_dataset)
    print("Size dataset:",training_dataset.x_size)
    print("Variable size dataset:",training_dataset.variable_size)

    X_static = np.tile(X_static.numpy().astype("float32"), (X.shape[0], 1, 1))
    X_static = X_static.reshape(X_static.shape[0]*X_static.shape[1], X_static.shape[2])
    X = X.numpy().astype("float32").reshape(X.shape[0]*X.shape[1], X.shape[2])
    Y_prog = Y_prog.numpy().astype("float32").reshape(Y_prog.shape[0]*Y_prog.shape[1], Y_prog.shape[2])
    Y_inc = Y_inc.numpy().astype("float32").reshape(Y_inc.shape[0]*Y_inc.shape[1], Y_inc.shape[2])

    print("Statics shape:", X_static.shape)
    print("Statics shape:", X.shape)
    print("Statics shape:", Y_prog.shape)
    print("Statics shape:", Y_inc.shape)
    print("")
    
    feats_train = np.concatenate((X_static, X, Y_prog), axis=-1)
    target_train = Y_inc
    
    del X_static, X, Y_prog, Y_inc
    
    X_static, X, Y_prog, Y_inc = validation_dataset.load_data()
    
    X_static = np.tile(X_static.numpy().astype("float32"), (X.shape[0], 1, 1))
    X_static = X_static.reshape(X_static.shape[0]*X_static.shape[1], X_static.shape[2])
    X = X.numpy().astype("float32").reshape(X.shape[0]*X.shape[1], X.shape[2])
    Y_prog = Y_prog.numpy().astype("float32").reshape(Y_prog.shape[0]*Y_prog.shape[1], Y_prog.shape[2])
    Y_inc = Y_inc.numpy().astype("float32").reshape(Y_inc.shape[0]*Y_inc.shape[1], Y_inc.shape[2])

    feats_val = np.concatenate((X_static, X, Y_prog), axis=-1)
    target_val = Y_inc
    
    del X_static, X, Y_prog, Y_inc
    
    # Setup the xgboost model instance and choose some parameters to control the training
    xgb_model = xgb.XGBRegressor(
        n_estimators=256,
        tree_method="hist", # enable computation on GPU if available
        # multi_strategy="multi_output_tree",
        # objective = 'reg:pseudohubererror',
        learning_rate=0.15,
        max_depth = 10,
        #reg_lambda= 0.001,
        eval_metric=r2_score_multi,
        subsample=0.7,
    )

    print("Fitting XGB model...")
    start_time = time.time()
    
    xgb_model.fit(feats_train, target_train, eval_set=[(feats_val, target_val)])
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Time required for fitting: ")
    print("--- %s seconds ---" % (elapsed_time))
    print("--- %s minutes ---" % (elapsed_time/60))

    xgb_model.save_model(os.path.join(path_to_results, 'xgb_model.bin'))
    
    print("XGB test coefficient of determination : ", xgb_model.score(feats_val, target_val))
    print("XGB train coefficient of determination : ", xgb_model.score(feats_train, target_train))

    #plot_feature_importances(xgb_model, dataset.feat_lst, save_to=os.path.join(path_to_results, 'feature_importance_xgb.pdf'))


if __name__ == "__main__":
    
    training_data = EcDataset(CONFIG,
                        CONFIG["start_year"],
                        CONFIG["end_year"])
    
    validation_data = EcDataset(CONFIG,
                        CONFIG["validation_start"],
                        CONFIG["validation_end"])
    
    train_xgb(training_data, validation_data, path_to_results)