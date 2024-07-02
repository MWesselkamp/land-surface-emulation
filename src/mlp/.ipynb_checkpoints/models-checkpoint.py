import torch
import torch.cuda
import torch.nn as nn
import numpy as np
import zarr

import pytorch_lightning as L
import torchmetrics

from torch import tensor

try:
    from mlp_2D.ec_database import TorchStandardScalerFeatTens
    from utils.utils import r2_score_multi
except ModuleNotFoundError:
    from src.mlp_2D.ec_database import TorchStandardScalerFeatTens
    from src.utils.utils import r2_score_multi
from pytorch_lightning.utilities import grad_norm


class MLPregressor(L.LightningModule):

    def __init__(self, input_size, hidden_size, output_size, batch_size, learning_rate, lookback, rollout, dropout, weight_decay, device,
                 loss = 'mse', activation = nn.ReLU(), db_path = '/perm/daep/ai_land_examples/ec-land_deltax_norm_mean_std.zarr'):

        super().__init__()
        self.save_hyperparameters(ignore=['criterion', 'activation'])
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.num_layers = len(hidden_size)
        self.activation = activation
        self.learning_rate = learning_rate
        self.output_size = output_size
        self.lookback = lookback
        self.rollout = rollout
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.criterion = nn.MSELoss() if loss == 'mse' else nn.SmoothL1Loss()
        self.my_device = device
        
        self.val_acc = torchmetrics.MeanAbsoluteError()
        self.test_acc = torchmetrics.MeanAbsoluteError()
        self.network = self.dense()

        self.targ_lst = ['swvl1',
                         'swvl2',
                         'swvl3',
                         'stl1',
                         'stl2',
                         'stl3',
                         'snowc',
                         ]

        self.targ_scalar = TorchStandardScalerFeatTens(
            path=db_path,
            feat_lst=self.targ_lst,
            dev=self.my_device,
            )

        print("Device: ", self.device)

        self.targ_idx_full = np.array([24, 25, 26, 27, 28, 29, 30])

    def dense(self):
        layers = nn.Sequential()
        layers.add_module(f'input', nn.Linear(self.input_size, self.hidden_size[0]))
        layers.add_module(f'activation0', self.activation)
        layers.add_module(f'dropout0', nn.Dropout(self.dropout))
        if self.num_layers > 1:
            for i in range(self.num_layers-1):
                layers.add_module(f'hidden{i}', nn.Linear(self.hidden_size[i], self.hidden_size[i + 1]))
                layers.add_module(f'activation{i}', self.activation)
                layers.add_module(f'dropout{i}', nn.Dropout(self.dropout))
        layers.add_module('output', nn.Linear(self.hidden_size[-1], self.output_size))
        
        return layers

    def forward(self, x):

        #x = x[:,:self.lookback,:,:]
        #print(x.size())
        prediction = self.network(x)

        return prediction

    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'val_loss'}

    def training_step(self, train_batch, batch_idx):

        x, y = train_batch  # x with lookback, y without -> len(x) > len(y)
        
        logits = self.forward(x)
        loss = self.criterion(self.targ_scalar.transform(logits),
                             self.targ_scalar.transform(y))
        self.log('train_loss_logit', loss, on_step=False, on_epoch=True, sync_dist=True)
        #train_loss = torch.zeros(1, dtype=x.dtype, device=self.device, requires_grad=False)
 
        x_rollout = x.clone()
        y_rollout = y.clone()

        # iterate over lead time.
        for step in range(self.rollout):
            # x = [batch_size=8, lookback (7) + rollout (3) = 10, n_feature = 37]
            x0 = x_rollout[:, step, :, :].clone()  # select input with lookback.
            y_hat = self.forward(x0)  # prediction at rollout step
            if step < self.rollout-1:
                x_rollout[:,step + 1,:,self.targ_idx_full] = x_rollout[:, step , :, self.targ_idx_full].clone() + y_hat
            y_rollout[:, step, :, :] = y_hat  # target at next step, that is 0 at future y
            
        train_step_loss = self.criterion(self.targ_scalar.transform(y_rollout),
                                self.targ_scalar.transform(y))
                
        # Compute loss for each variable separately and accumulate
        variable_losses = torch.zeros(len(self.targ_idx_full), dtype=x.dtype, device=self.device, requires_grad=False)
        for idx, var_idx in enumerate(self.targ_idx_full):
            var_loss = self.criterion(self.targ_scalar.transform(y_rollout)[:, :, :, idx],
                                  self.targ_scalar.transform(y)[:, :, :, idx])
            variable_losses[idx] = var_loss

        #variable_losses /= self.rollout
        train_step_loss /= self.rollout  # average rollout loss

        self.log('train_loss_step', train_step_loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log('train_loss', loss + train_step_loss, on_step=False, on_epoch=True, sync_dist=True)
        for idx, v_loss in enumerate(variable_losses):
            self.log(f'train_loss_var_{idx}', v_loss, on_step=False, on_epoch=True, sync_dist=True)

        return loss + train_step_loss

    def validation_step(self, val_batch, batch_idx):

        x, y = val_batch  # IF x with lookback, y without -> len(x) > len(y)
        
        logits = self.forward(x)
        loss = self.criterion(self.targ_scalar.transform(logits),
                             self.targ_scalar.transform(y))
        self.log('val_loss_logit', loss, on_step=False, on_epoch=True, sync_dist=True)
   
        #val_loss = torch.zeros(1, dtype=x.dtype, device=self.device, requires_grad=False)
        x_rollout = x.clone()
        y_rollout = y.clone()

        # iterate over lead time.
        for step in range(self.rollout):
            x0 = x_rollout[:, step, :, :].clone()  # select input with lookback.
            y_hat = self.forward(x0)  # prediction at rollout step
            if step < self.rollout - 1:
                x_rollout[:, step + 1, :, self.targ_idx_full] = x_rollout[:, step , :,
                                                               self.targ_idx_full].clone() + y_hat
            y_rollout[:, step, :, :] = y_hat  # target at next step, that is 0 at future y
        
        val_step_loss = self.criterion(self.targ_scalar.transform(y_rollout),
                                    self.targ_scalar.transform(y))

        # Compute loss for each variable separately and accumulate
        variable_losses = torch.zeros(len(self.targ_idx_full), dtype=x.dtype, device=self.device, requires_grad=False)
        for idx, var_idx in enumerate(self.targ_idx_full):
            var_loss = self.criterion(self.targ_scalar.transform(y_rollout)[:, :, :,idx],
                                  self.targ_scalar.transform(y)[:, :, :, idx])
            variable_losses[idx] = var_loss

        val_step_loss /= self.rollout
        #val_loss /= self.rollout  # average rollout loss

        print("Loss:", val_step_loss)

        r2 = r2_score_multi(self.targ_scalar.transform(y_rollout).cpu(),
                            self.targ_scalar.transform(y).cpu())

        self.log('val_loss_step', val_step_loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log('val_loss', loss + val_step_loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log('val_r**2', r2, on_step=False, on_epoch=True, sync_dist=True)
        for idx, v_loss in enumerate(variable_losses):
            self.log(f'val_loss_var_{idx}', v_loss, on_step=False, on_epoch=True, sync_dist=True)

        return loss + val_step_loss

    def forecast(self, static_feats, dynamic_feats):
        
        lead_time = dynamic_feats.shape[0]
        
        self.eval()
        for t in range(lead_time - 1):
            if t % 1000 == 0:
                print(f"on step {t}...")
            with torch.no_grad():
                logits = self.forward(torch.cat((static_feats, dynamic_feats[[t]]), axis=-1))
                preds = dynamic_feats[t, :, -len(self.targ_lst):].clone() + logits.squeeze()
                dynamic_feats[t + 1, :, -len(self.targ_lst):] = preds
                
        return dynamic_feats

class MLPregressor_global(L.LightningModule):

    """
    See also: https://github.com/da-ewanp/ai-land/blob/main/ai_land/model.py
    """
    
    def __init__(self, input_clim_dim,
                 input_met_dim,
                 input_state_dim,
                 hidden_dim,
                 output_dim,
                 output_diag_dim,
                 batch_size,
                 learning_rate,
                 lookback,
                 rollout,
                 dropout, 
                 weight_decay,
                 device,
                 loss = 'mse', 
                 activation = nn.ReLU(), 
                 targets = ['swvl1',
                         'swvl2',
                         'swvl3',
                         'stl1',
                         'stl2',
                         'stl3',
                         'snowc',
                         ],
                 db_path = ''):

        super().__init__()
        
        self.my_device = device
        
        ds = zarr.open(db_path)
        fistdiff_idx = [list(ds["variable"]).index(x) for x in targets]
        self.ds_data_std = tensor(ds.data_stdevs[fistdiff_idx]).to(device = self.my_device)
        self.ds_mean = tensor(ds.data_1stdiff_means[fistdiff_idx]).to(device = self.my_device)/self.ds_data_std
        self.ds_std = tensor(ds.data_1stdiff_stdevs[fistdiff_idx]).to(device = self.my_device)/self.ds_data_std
        
        self.save_hyperparameters(ignore=['criterion', 'activation'])
        
        self.input_clim_dim = input_clim_dim
        self.input_met_dim = input_met_dim
        self.input_state_dim = input_state_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.output_diag_dim = output_diag_dim
        self.batch_size = batch_size
        self.num_layers = len(hidden_dim)
        self.activation = activation
        self.learning_rate = learning_rate
        self.output_dim = output_dim
        self.lookback = lookback
        self.rollout = rollout
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.criterion = nn.MSELoss() if loss == 'mse' else nn.SmoothL1Loss()
        self.targ_lst = targets

        self.val_acc = torchmetrics.MeanAbsoluteError()
        self.test_acc = torchmetrics.MeanAbsoluteError()
        
        self.network = self.dense()  

        print("Device: ", self.device)

        #self.targ_idx_full = np.array([24, 25, 26, 27, 28, 29, 30])
        
    def transform(self, x, mean, std):
        x_norm = (x - mean) / (std + 1e-5)
        return x_norm

    def dense(self):
        layers = nn.Sequential()
        layers.add_module(f'input', nn.Linear(self.input_clim_dim+self.input_met_dim+self.input_state_dim, self.hidden_dim[0]))
        layers.add_module(f'activation0', self.activation)
        layers.add_module(f'dropout0', nn.Dropout(self.dropout))
        if self.num_layers > 1:
            for i in range(self.num_layers-1):
                layers.add_module(f'hidden{i}', nn.Linear(self.hidden_dim[i], self.hidden_dim[i + 1]))
                layers.add_module(f'activation{i}', self.activation)
                layers.add_module(f'dropout{i}', nn.Dropout(self.dropout))
        layers.add_module('output', nn.Linear(self.hidden_dim[-1], self.output_dim))
        
        return layers

    def forward(self, x_clim, x_met, x_state):

        x = torch.cat((x_clim, x_met, x_state), dim = -1)
        prediction = self.network(x)

        return prediction

    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'val_loss'}

    def training_step(self, train_batch, batch_idx):

        x_clim, x_met, x_state, y = train_batch
        
        logits = self.forward(x_clim, x_met, x_state)
        loss = self.criterion(self.transform(logits, self.ds_mean, self.ds_std),
                             self.transform(y, self.ds_mean, self.ds_std))
        self.log('train_loss_logit', loss, on_step=False, on_epoch=True, sync_dist=True)
        #train_loss = torch.zeros(1, dtype=x.dtype, device=self.device, requires_grad=False)
 
        x_state_rollout = x_state.clone()
        y_rollout = y.clone()
        #y_rollout_diag = y_diag.clone()

        # iterate over lead time.
        for step in range(self.rollout):
            # x = [batch_size=8, lookback (7) + rollout (3) = 10, n_feature = 37]
            x0 = x_state_rollout[:, step, :, :].clone()  # select input with lookback.
            y_hat = self.forward(x_clim[:, step, :, :], x_met[:, step, :, :], x0)  # prediction at rollout step
            y_rollout[:, step, :, :] = y_hat   # overwrite y with prediction.
            if step < self.rollout-1:
                x_state_rollout[:,step + 1,:, :] = x_state_rollout[:, step , :, :].clone() + y_hat # target at next step
            
            
        train_step_loss = self.criterion(self.transform(y_rollout, self.ds_mean, self.ds_std),
                                self.transform(y, self.ds_mean, self.ds_std))
                
        # Compute loss for each variable separately and accumulate
        variable_losses = torch.zeros(len(self.targ_lst), dtype=y.dtype, device=self.device, requires_grad=False)
        for idx, var_idx in enumerate(self.targ_lst):
            var_loss = self.criterion(self.transform(y_rollout, self.ds_mean, self.ds_std)[:, :, :, idx],
                                  self.transform(y, self.ds_mean, self.ds_std)[:, :, :, idx])
            variable_losses[idx] = var_loss

        #variable_losses /= self.rollout
        train_step_loss /= self.rollout  # average rollout loss

        self.log('train_loss_step', train_step_loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log('train_loss', loss + train_step_loss, on_step=False, on_epoch=True, sync_dist=True)
        for idx, v_loss in enumerate(variable_losses):
            self.log(f'train_loss_var_{idx}', v_loss, on_step=False, on_epoch=True, sync_dist=True)

        return loss + train_step_loss

    def validation_step(self, val_batch, batch_idx):

        x_clim, x_met, x_state, y = val_batch  # IF x with lookback, y without -> len(x) > len(y)
        
        logits = self.forward(x_clim, x_met, x_state)
        loss = self.criterion(self.transform(logits, self.ds_mean, self.ds_std),
                             self.transform(y, self.ds_mean, self.ds_std))
        self.log('val_loss_logit', loss, on_step=False, on_epoch=True, sync_dist=True)
   
        x_state_rollout = x_state.clone()
        y_rollout = y.clone()
        #y_rollout_diag = y_diag.clone()

        # iterate over lead time.
        for step in range(self.rollout):
            # x = [batch_size=8, lookback (7) + rollout (3) = 10, n_feature = 37]
            x0 = x_state_rollout[:, step, :, :].clone()  # select input with lookback.
            y_hat = self.forward(x_clim[:, step, :, :], x_met[:, step, :, :], x0)  # prediction at rollout step
            y_rollout[:, step, :, :] = y_hat   # overwrite y with prediction.
            if step < self.rollout-1:
                x_state_rollout[:,step + 1,:, :] = x_state_rollout[:, step , :, :].clone() + y_hat # target at next step
            
            
        val_step_loss = self.criterion(self.transform(y_rollout, self.ds_mean, self.ds_std),
                                self.transform(y, self.ds_mean, self.ds_std))
        val_step_loss /= self.rollout # average rollout loss
                
        # Compute loss for each variable separately and accumulate
        variable_losses = torch.zeros(len(self.targ_lst), dtype=y.dtype, device=self.device, requires_grad=False)
        for idx, var_idx in enumerate(self.targ_lst):
            var_loss = self.criterion(self.transform(y_rollout, self.ds_mean, self.ds_std)[:, :, :, idx],
                                  self.transform(y, self.ds_mean, self.ds_std)[:, :, :, idx])
            variable_losses[idx] = var_loss

        print("Loss:", val_step_loss)

        r2 = r2_score_multi(self.transform(y_rollout, self.ds_mean, self.ds_std).cpu(),
                            self.transform(y, self.ds_mean, self.ds_std).cpu())

        self.log('val_loss_step', val_step_loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log('val_loss', loss + val_step_loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log('val_r**2', r2, on_step=False, on_epoch=True, sync_dist=True)
        for idx, v_loss in enumerate(variable_losses):
            self.log(f'val_loss_var_{idx}', v_loss, on_step=False, on_epoch=True, sync_dist=True)

        return loss + val_step_loss

    def forecast(self, x_clim, x_met, x_state):
        
        preds = states.clone().to(self.device)
        lead_time = x_state.shape[0]
        
        self.eval()
        
        for t in range(lead_time - 1):
            
            if t % 1000 == 0:
                print(f"on step {t}...")
                
            with torch.no_grad():
                logits = self.forward(torch.cat((x_clim, x_met[[t]], preds[[t]]), axis=-1))
                preds[t+1, ...] = preds[t, ...] + logits
                
        return preds
