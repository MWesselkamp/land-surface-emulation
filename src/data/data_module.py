import os
import sys 
import xarray as xr
import random

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

from typing import Tuple

import cftime
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import yaml
import zarr
from torch import tensor
from torch.utils.data import DataLoader, Dataset

#PATH = '../data'
## Open up experiment config
#with open(os.path.join(PATH, "config.yaml")) as stream:
#    try:
#        CONFIG = yaml.safe_load(stream)
#    except yaml.YAMLError as exc:
#        print(exc)

torch.cuda.empty_cache()

class EcDataset(Dataset):
    # load the dataset
    def __init__(
        self, 
        config,
        start_yr,
        end_yr
    ):
        
        x_idxs=config["x_slice_indices"]
        
        self.min_spatial_sample = config["spatial_sample_size"]
            
        path=config["file_path"]
        
        self.rollout=config["roll_out"]
        self.lookback=config["lookback"]
        self.model = config["model"]

        self.dyn_transform, self.dyn_inv_transform = self.select_transform(config["dyn_transform"])
        self.stat_transform, self.stat_inv_transform = self.select_transform(config["stat_transform"])
        self.prog_transform, self.prog_inv_transform = self.select_transform(config["prog_transform"])
        self.diag_transform, self.diag_inv_transform = self.select_transform(config["diag_transform"])

        self.ds_ecland = zarr.open(path)
        # Create time index to select appropriate data range
        date_times = pd.to_datetime(
            cftime.num2pydate(
                self.ds_ecland["time"], self.ds_ecland["time"].attrs["units"]
            )
        )
        self.start_index = min(np.argwhere(date_times.year == int(start_yr)))[0]
        self.end_index = max(np.argwhere(date_times.year == int(end_yr)))[0]
        print("Start index", self.start_index)
        if self.model == 'lstm':
            self.start_index = self.start_index - self.lookback
            print("Start index", self.start_index)
        self.times = np.array(date_times[self.start_index : self.end_index])
        self.len_dataset = self.end_index - self.start_index
        print("Length of dataset:", self.len_dataset)

        # Select points in space
        print("Use all x_idx from global.")
        self.x_idxs = (0, None) if "None" in x_idxs else x_idxs
        self.x_size = len(self.ds_ecland["x"][slice(*self.x_idxs)])
        self.lats = self.ds_ecland["lat"][slice(*self.x_idxs)]
        self.lons = self.ds_ecland["lon"][slice(*self.x_idxs)]

        # If spatial sampling is active, create container with random indices.
        if self.min_spatial_sample is not None:
            print("Activate spatial sampling of x_idxs")
            self.spatial_sample_size = self.find_spatial_sample_size(self.min_spatial_sample)
            print("Spatial sample size:", self.spatial_sample_size)
            self.chunked_x_idxs = self.chunk_indices(self.spatial_sample_size)
            self.chunk_size = len(self.chunked_x_idxs)
            print("Chunk size:", self.chunk_size)
        else:
            print("Use all x_idx from global.")
            self.spatial_sample_size = None

        # List of climatological time-invariant features
        self.static_feat_lst = config["clim_feats"]
        self.clim_index = [
            list(self.ds_ecland["clim_variable"]).index(x) for x in config["clim_feats"]
        ]
        # List of features that change in time
        self.dynamic_feat_lst = config["dynamic_feats"]
        self.dynamic_index = [
            list(self.ds_ecland["variable"]).index(x) for x in config["dynamic_feats"]
        ]
        # Prognostic target list
        self.targ_lst = config["targets_prog"]
        self.targ_index = [
            list(self.ds_ecland["variable"]).index(x) for x in config["targets_prog"]
        ]
        # Diagnostic target list
        self.targ_diag_lst = config["targets_diag"]
        if self.targ_diag_lst is not None:
            self.targ_diag_index = [
                list(self.ds_ecland["variable"]).index(x) for x in config["targets_diag"]
            ]
            self.y_diag_means = tensor(self.ds_ecland.data_means[self.targ_diag_index])
            self.y_diag_stdevs = tensor(self.ds_ecland.data_stdevs[self.targ_diag_index])
        else:
            self.targ_diag_index = None
        
        self.variable_size = len(self.dynamic_index) + len(self.targ_index ) + len(self.clim_index)
        
        # Define the statistics used for normalising the data
        self.x_dynamic_means = tensor(self.ds_ecland.data_means[self.dynamic_index])
        self.x_dynamic_stdevs = tensor(self.ds_ecland.data_stdevs[self.dynamic_index])
        self.x_dynamic_maxs = tensor(self.ds_ecland.data_maxs[self.dynamic_index])
        
        self.clim_means = tensor(self.ds_ecland.clim_means[self.clim_index])
        self.clim_stdevs = tensor(self.ds_ecland.clim_stdevs[self.clim_index])
        self.clim_maxs = tensor(self.ds_ecland.clim_maxs[self.clim_index])
        
        # Define statistics for normalising the targets
        self.y_prog_means = tensor(self.ds_ecland.data_means[self.targ_index])
        self.y_prog_stdevs = tensor(self.ds_ecland.data_stdevs[self.targ_index])
        self.y_prog_maxs = tensor(self.ds_ecland.data_maxs[self.targ_index])

        # Create time-invariant static climatological features
        x_static = tensor(
            self.ds_ecland.clim_data[slice(*self.x_idxs), :]
        )
        x_static = x_static[:, self.clim_index]
        self.x_static_scaled = self.stat_transform(
            x_static, means = self.clim_means, stds = self.clim_stdevs, maxs = self.clim_maxs
        ).reshape(1, self.x_size, -1)

    def chunk_indices(self, chunk_size = 2000):

        indices = list(range(self.x_size))
        random.shuffle(indices)
        
        spatial_chunks = [indices[i:i + chunk_size] for i in range(0, self.x_size, chunk_size)]
        
        return spatial_chunks

    def find_spatial_sample_size(self, limit):

        for i in range(limit, self.x_size):
            if self.x_size % i == 0:
                return i

    def select_transform(self, transform_spec):

        if transform_spec == "zscoring":
            self.transform = self.z_transform
            self.inv_transform = self.inv_z_transform
        elif transform_spec == "max":
            self.transform = self.max_transform
            self.inv_transform = self.inv_max_transform
        elif transform_spec == "identity":
            self.transform = self.id_transform
            self.inv_transform = self.inv_id_transform
            
        return self.transform, self.inv_transform
            
    def id_transform(self, x: np.ndarray, **kwargs) -> np.ndarray:
        """Placeholder identity function

        :param x: data :param mean: mean :param std: standard deviation :return:
        normalised data
        """
        return x

    def inv_id_transform(
        self, x: np.ndarray, **kwargs
    ) -> np.ndarray:
        """Placeholder inverse identity function.

        :param x_norm: normlised data :param mean: mean :param std: standard deviation
        :return: unnormalised data
        """
        return x

    def z_transform(self, x: np.ndarray, **kwargs) -> np.ndarray:
        """Transform data with mean and stdev.

        :param x: data :param mean: mean :param std: standard deviation :return:
        normalised data
        """
        x_norm = (x - kwargs["means"]) / (kwargs["stds"] + 1e-5)
        return x_norm

    def inv_z_transform(
        self, x_norm: np.ndarray, **kwargs
    ) -> np.ndarray:
        """Inverse transform on data with mean and stdev.

        :param x_norm: normlised data :param mean: mean :param std: standard deviation
        :return: unnormalised data
        """
        x = (x_norm * (kwargs["stds"] + 1e-5)) + kwargs["means"]
        return x
    
    def max_transform(self, x: np.ndarray, **kwargs) -> np.ndarray:
        """Transform data with max.

        :param x: data :param mean: mean :param std: standard deviation :return:
        normalised data
        """
        x_norm = (x / kwargs["maxs"])  # + 1e-5
        return x_norm
    
    def inv_max_transform(self, x_norm: np.ndarray, **kwargs) -> np.ndarray:
        """Transform data with max.

        :param x: data :param mean: mean :param std: standard deviation :return:
        normalised data
        """
        x = (x * kwargs["maxs"])  # + 1e-5
        return x

    def load_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load data into memory. **CAUTION ONLY USE WHEN WORKING WITH DATASET THAT FITS
        IN MEM**

        :return: static_features, dynamic_features, prognostic_targets,
        diagnostic_targets
        """
        
        ds_slice = tensor(
            self.ds_ecland.data[
                self.start_index : self.end_index, slice(*self.x_idxs), :
            ]
        )

        X = ds_slice[:, :, self.dynamic_index]
        X = self.dyn_transform(X, means = self.x_dynamic_means, stds = self.x_dynamic_stdevs, maxs = self.x_dynamic_maxs)

        X_static = self.x_static_scaled

        Y_prog = ds_slice[:, :, self.targ_index]
        Y_prog = self.prog_transform(Y_prog, means = self.y_prog_means, stds = self.y_prog_stdevs, maxs = self.y_prog_maxs)

        if self.model == 'xgb':
            
            Y_inc = Y_prog[1:, :, :] - Y_prog[:-1, :, :]
            
            return X_static, X[:-1], Y_prog[:-1], Y_inc

        else:
            
            if self.targ_diag_index is not None:
                
                Y_diag = ds_slice[:, :, self.targ_diag_index]
                Y_diag = self.diag_transform(Y_diag, means = self.y_diag_means, stds = self.y_diag_stdevs,  maxs = self.y_diag_maxs)
    
                return X_static, X, Y_prog, Y_diag
    
            else:
                
                return X_static, X, Y_prog
            
    # number of rows in the dataset
    def __len__(self):
        
        return ((self.len_dataset - 1 - self.rollout) * self.chunk_size) if self.spatial_sample_size is not None else ((self.len_dataset - 1 - self.rollout) * self.x_size) 

    # get a row at an index
    def __getitem__(self, idx):

        # print(f"roll: {self.rollout}, lends: {self.len_dataset}, x_size: {self.x_size}")
         
        t_start_idx = (idx % (self.len_dataset - 1 - self.rollout)) + self.start_index
        t_end_idx = (idx % (self.len_dataset - 1 - self.rollout)) + self.start_index + self.lookback + self.rollout + 1
        
        if self.spatial_sample_size is not None:
            x_idx = [x + self.x_idxs[0] for x in self.chunked_x_idxs[(idx % self.chunk_size)]]
        else:
            x_idx = (idx % self.x_size) + self.x_idxs[0]
        
        ds_slice = tensor(
            self.ds_ecland.data[
                slice(t_start_idx, t_end_idx), :, :
            ]
        )
        print("x_idx:", x_idx)
        print("ds_slice shape:", ds_slice.shape)
        ds_slice = ds_slice[:, x_idx, :]
        print("ds_slice shape:", ds_slice.shape)

        X = ds_slice[:, :, self.dynamic_index]
        X = self.dyn_transform(X, means = self.x_dynamic_means, stds = self.x_dynamic_stdevs, maxs = self.x_dynamic_maxs)
        
        X_static = self.x_static_scaled.expand(self.rollout+self.lookback, -1, -1)
        X_static = X_static[:, x_idx, :]
        
        Y_prog = ds_slice[:, :, self.targ_index]
        Y_prog = self.prog_transform(Y_prog, means = self.y_prog_means, stds = self.y_prog_stdevs, maxs = self.y_prog_maxs)

        # Calculate delta_x update for corresponding x state
        Y_inc = Y_prog[1:,:, :] - Y_prog[:-1, :, :]
        
        if self.model == 'mlp':
            
            if self.targ_diag_index is not None:
            
                Y_diag = ds_slice[:, :, self.targ_diag_index]
                Y_diag = self.diag_transform(Y_diag,  means = self.y_diag_means, stds = self.y_diag_stdevs,  maxs = self.y_diag_maxs)
                Y_diag = Y_diag[:-1]
                
                return X_static, X[:-1], Y_prog[:-1], Y_inc, Y_diag
            else:
                return X_static, X[:-1], Y_prog[:-1], Y_inc
        
        elif self.model == 'lstm':

            X_static_h = X_static[:self.lookback]
            X_static_f = X_static[self.lookback:(self.lookback + self.rollout)]
            
            X_h = X[:self.lookback]
            X_f = X[self.lookback:(self.lookback + self.rollout)]
        
            Y_prog_h = Y_prog[:self.lookback]
            Y_prog_f = Y_prog[self.lookback:(self.lookback + self.rollout)]

            Y_inc_h = Y_inc[:self.lookback]
            Y_inc_f = Y_inc[self.lookback:(self.lookback + self.rollout)]
            
            if self.targ_diag_index is not None:
                
                Y_diag = ds_slice[:, :, self.targ_diag_index]
                Y_diag = self.transform(Y_diag, self.y_diag_means, self.y_diag_stdevs)
                
                Y_diag_h = Y_diag[:self.lookback]
                Y_diag_f = Y_diag[self.lookback:self.lookback + self.rollout]

                return X_static_h, X_static_f, X_h, X_f, Y_prog_h, Y_prog_f, Y_diag_h, Y_diag_f
            
            else:
                
                return X_static_h, X_static_f, X_h, X_f, Y_prog_h, Y_prog_f, Y_inc_h, Y_inc_f
            
class NonLinRegDataModule(pl.LightningDataModule):
    """Pytorch lightning specific data class."""
    
    def __init__(self, config):
        
        super(NonLinRegDataModule, self).__init__()
        
        self.config = config
        

    def setup(self, stage):
        # generator = torch.Generator().manual_seed(42)
        self.train = EcDataset(self.config, 
                               start_yr=self.config["start_year"], 
                               end_yr=self.config["end_year"])
        self.test = EcDataset(self.config, 
                              start_yr=self.config["validation_start"], 
                              end_yr=self.config["validation_end"]
        )
        
        self.preds = EcDataset(self.config, 
                              start_yr=self.config["validation_start"], 
                              end_yr=self.config["validation_end"],
        #                      spatial_splits=self.config["spatial_splits"],
        )

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.config["batch_size"],
            shuffle=True,
            drop_last=True,
            num_workers=self.config["num_workers"],
            persistent_workers=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.test,
            batch_size=self.config["batch_size"],
            shuffle=False,
            drop_last = True,
            num_workers=self.config["num_workers"],
            persistent_workers=True,
            pin_memory=True,
        )
    
    def pred_dataloader(self):
        
        return DataLoader(
            self.preds,
            batch_size=1,
            shuffle=False,
            drop_last = True,
            num_workers=self.config["num_workers"],
            persistent_workers=True,
            pin_memory=True,
        )