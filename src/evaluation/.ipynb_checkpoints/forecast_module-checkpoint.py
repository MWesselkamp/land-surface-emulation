import numpy as np
import os
import pandas as pd
import pytorch_lightning as pl
import torch
import random
import xarray as xr
import re
import xgboost as xgb
import time
import torch

from torch import tensor

if torch.cuda.is_available():
    DEVICE = "cuda:0"
else:
    DEVICE = "cpu"
#DEVICE = "cpu" #torch.device(dev)
#print(DEVICE)

torch.cuda.empty_cache()

class ForecastModule:
    
    def __init__(self, model, my_device = None):
        
        if 'MLP' in type(model).__name__:
            print("Instance is mlp model")
            self.filename = 'mlp'
        elif 'LSTM' in type(model).__name__:
            print("Instance is lstm model")
            self.filename = 'lstm'
        elif type(model).__name__ == 'Booster':
            print("Instance is XGBModel")
            self.filename = 'xgb'
        else:
            print("Model type is:", type(model))
            print("Don't know model type")
        
        self.model = model

        if my_device is None:
            self.my_device = DEVICE
        else:
            self.my_device = my_device
        
    def get_test_data_europe(self, dataset, transform = 'zscoring'):
        
        if isinstance(self.model, pl.LightningModule):
            
            feats = dataset.ds_ecland[dataset.dynamic_feat_lst].isel(time=slice(0, -1)).compute() #(500-4, 500+4)
            dynamic_features = np.array(feats.to_array().values.T)
            
            if transform == 'zscoring':
                dynamic_features = dataset.dynamic_feat_scalar.transform(tensor(dynamic_features, dtype=torch.float32)).transpose(0, 1)
            elif transform == 'log':
                print("apply log transform")
                dynamic_features = torch.log(tensor(dynamic_features, dtype=torch.float32).transpose(0, 1) + 1) 
                
            static_features = dataset.X_static_scaled
            
        elif isinstance(self.model, xgb.Booster):
            
            feats = dataset.ds_ecland[dataset.dynamic_feat_lst].isel(time=slice(0, -1)).compute() #(500-4, 500+4)
            dynamic_features = np.array(feats.to_array().values.T).transpose(1,0,2)
            static_features = np.expand_dims(dataset.X_static, axis=0)

        print(static_features.shape)
        print(dynamic_features.shape)
        
        self.dataset = dataset
        self.scale = 'europe'
        
        return static_features, dynamic_features
    
    def get_test_data_global(self, dataset):

        if self.filename == 'xgb':
            X_static, X_met, Y_prog, Y_inc = dataset.load_data()  
        else:
             X_static, X_met, Y_prog = dataset.load_data() 
            
        self.dataset = dataset
        
        return X_static, X_met, Y_prog

    def get_test_data_chunk(self, dataset, chunk_idx = (0, None)):

        dataset.x_idxs = chunk_idx
        
        if self.filename == 'xgb':
            X_static, X_met, Y_prog, Y_inc = dataset.load_data()  
        else:
             X_static, X_met, Y_prog = dataset.load_data() 
            
        self.dataset = dataset
        
        return X_static, X_met, Y_prog

    def get_climatology(self, file_path, variability = False):

        clim_stats = 'clim_6hr_std' if variability else 'clim_6hr_mu'
        
        if file_path is not None:
            
            climatology = xr.open_dataset(file_path)
            clim_array = climatology[clim_stats].sel(variable=self.dataset.targ_lst).values
            clim_array = np.tile(clim_array, (2, 1, 1))

        else:
            clim_array = None

        return clim_array
        
    def step_forecast(self, static_features, dynamic_features, time_idx_0, targ_lst, inv_transform = 'zscoring'):
        
        if isinstance(self.model, pl.LightningModule):
            print("Setting model to evaluation mode")
            dynamic_features_prediction = dynamic_features.clone()
            self.model.eval()
        elif isinstance(self.model, xgb.Booster):
            dynamic_features_prediction = dynamic_features.copy()
        else:
            print("Don't know model type.")
        
        start_time = time.time()
            
        if self.filename == 'mlp':
            with torch.no_grad():
                
                for time_idx in range(time_idx_0, dynamic_features.shape[0]-1):
                    
                    if time_idx % 1000 == 0:
                        print(f"on step {time_idx}...")

                    step_predictors = torch.cat((static_features, dynamic_features_prediction[[time_idx]]), axis=-1)
                    logits = self.model.forward(step_predictors)
                    dynamic_features_prediction[time_idx+1, :, -len(targ_lst):] = dynamic_features_prediction[time_idx, :, -len(targ_lst):] + logits.squeeze()

        elif self.filename == 'xgb':
                
            for time_idx in range(time_idx_0, dynamic_features.shape[0]-1):
                
                if time_idx % 1000 == 0:
                    print(f"on step {time_idx}...")                        
                step_predictors = xgb.DMatrix(np.concatenate((static_features, dynamic_features_prediction[[time_idx]]), axis=-1).squeeze())
                logits = self.model.predict(step_predictors)
                dynamic_features_prediction[time_idx+1, :, -len(targ_lst):] = dynamic_features_prediction[time_idx, :, -len(targ_lst):] + logits.squeeze()
                
        elif self.filename == 'lstm':
            
            static_features, dynamic_features_prediction = static_features.to(self.my_device), dynamic_features_prediction.to(self.my_device)
            self.model.to(self.my_device)
            
            preds = self.model.forecast(static_features, dynamic_features_prediction)
            dynamic_features_prediction[self.model.lookback:,:,-len(targ_lst):] = preds.squeeze()

            dynamic_features_prediction = dynamic_features_prediction[self.model.lookback:, ...]
            dynamic_features = dynamic_features[self.model.lookback:, ...]

        end_time = time.time()
        elapsed_time = end_time - start_time
        print("--- %s seconds ---" % (elapsed_time))
        print("--- %s minutes ---" % (elapsed_time/60))
        
        self.dynamic_features_prediction = dynamic_features_prediction
        self.dynamic_features = dynamic_features
        
        if (self.filename == 'lstm') | (self.filename == 'mlp'):
                
            if inv_transform == 'zscoring':
                print("Backtransforming: zscoring")
                self.dynamic_features_prediction = self.dataset.dynamic_feat_scalar.inv_transform(self.dynamic_features_prediction.cpu()).numpy()
                self.dynamic_features = self.dataset.dynamic_feat_scalar.inv_transform(self.dynamic_features.cpu()).numpy() 
                
            elif inv_transform == 'log':
                print("Backtransforming : log")
                dynamic_features_prediction = np.exp(self.dynamic_features_prediction.cpu().numpy()) - 1
                dynamic_features = np.exp(self.dynamic_features.cpu().numpy()) - 1
                
        return self.dynamic_features, self.dynamic_features_prediction
    
    
    def step_forecast_global(self, X_static, X_met, Y_prog):
        
        if isinstance(self.model, pl.LightningModule):
            
            X_static, X_met, Y_prog = X_static.to(self.my_device), X_met.to(self.my_device), Y_prog.to(self.my_device)
            self.model.to(self.my_device)
        
            Y_prog_prediction = Y_prog.clone()
            
            print("Setting model to evaluation mode")
            self.model.eval()
            
        elif isinstance(self.model, xgb.Booster):
            Y_prog_prediction = Y_prog.numpy().copy()
        else:
            print("Don't know model type.")
        
        start_time = time.time()
        
        if self.filename == 'mlp':
            
            with torch.no_grad():
                
                for time_idx in range(Y_prog_prediction.shape[0]-1):
                    
                    if time_idx % 1000 == 0:
                        print(f"on step {time_idx}...")

                    logits = self.model.forward(X_static, X_met[[time_idx]], Y_prog_prediction[[time_idx]])
                    Y_prog_prediction[time_idx+1, ...] = Y_prog_prediction[time_idx, ...] + logits.squeeze()
            
        elif self.filename == 'lstm':

            print("Y_prog_prediction shape:", Y_prog_prediction.shape)
            preds = self.model.forecast(X_static, X_met, Y_prog)
            print("LSTM preds shape:", preds.shape)
            Y_prog_prediction[self.model.lookback:, ...] = preds.squeeze()

            Y_prog_prediction = Y_prog_prediction[self.model.lookback:, ...]
            Y_prog = Y_prog[self.model.lookback:, ...]

        elif self.filename == 'xgb':
                
            for time_idx in range(Y_prog_prediction.shape[0]-1):
                
                if time_idx % 10 == 0:
                    print(f"on step {time_idx}...")                        
                step_predictors = xgb.DMatrix(np.concatenate((X_static, X_met[[time_idx]], Y_prog_prediction[[time_idx]]), axis=-1).squeeze())
                logits = self.model.predict(step_predictors)
                Y_prog_prediction[time_idx+1, ...] = Y_prog_prediction[time_idx, ...] + logits.squeeze()


        end_time = time.time()
        elapsed_time = end_time - start_time
        print("--- %s seconds ---" % (elapsed_time))
        print("--- %s minutes ---" % (elapsed_time/60))

        print(Y_prog.shape)
        print(Y_prog_prediction.shape)
        
        self.dynamic_features = Y_prog
        self.dynamic_features_prediction = tensor(Y_prog_prediction) if self.filename == 'xgb' else Y_prog_prediction
                
        print("Backtransforming")
        self.dynamic_features_prediction = self.dataset.prog_inv_transform(self.dynamic_features_prediction.cpu(), means = self.dataset.y_prog_means, stds = self.dataset.y_prog_stdevs, maxs = self.dataset.y_prog_maxs).numpy()
        self.dynamic_features = self.dataset.prog_inv_transform(self.dynamic_features.cpu(), means = self.dataset.y_prog_means, stds = self.dataset.y_prog_stdevs, maxs = self.dataset.y_prog_maxs).numpy()

        return self.dynamic_features, self.dynamic_features_prediction

    def save_forecast(self, save_to):
        
        if self.scale == 'europe':
            
            if (self.filename == 'lstm') | (self.filename == 'mlp'):

                print("Backtransforming")
                self.dynamic_features_prediction = self.dataset.dynamic_feat_scalar.inv_transform(self.dynamic_features_prediction.cpu()).numpy()
                self.dynamic_features = self.dataset.dynamic_feat_scalar.inv_transform(self.dynamic_features.cpu()).numpy()

            obs_array = xr.DataArray(self.dynamic_features,
                                  coords={"time": self.dataset.full_times[:self.dynamic_features.shape[0]],
                                          "location": self.dataset.ds_ecland['x'],
                                          "variable": self.dataset.dynamic_feat_lst},
                                     dims=["time", "x", "variable"])
            obs_array.to_netcdf(os.path.join(save_to, f'fc_ecland_{self.filename}.nc'))
            print("saved reference data to: ", os.path.join(save_to, f'fc_ecland_{self.filename}.nc'))

            preds_array = xr.DataArray(self.dynamic_features_prediction,
                                  coords={"time": self.dataset.full_times[:self.dynamic_features_prediction.shape[0]],
                                          "location": self.dataset.ds_ecland['x'],
                                          "variable": self.dataset.dynamic_feat_lst},
                                      dims=["time", "x", "variable"])
            preds_array.to_netcdf(os.path.join(save_to, f'fc_ailand_{self.filename}.nc'))
            print("saved forecast to: ", os.path.join(save_to, f'fc_ecland_{self.filename}.nc'))

            return self.dynamic_features, self.dynamic_features_prediction
    
        else:
            
            print("Don't know scale to save data at.")