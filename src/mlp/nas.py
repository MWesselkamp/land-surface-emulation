import pytorch_lightning as L
import pandas as pd
import time
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
import torch
import csv
import sys
import os.path
import optuna

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from optuna.integration import PyTorchLightningPruningCallback
from mlp_2D.models import MLPregressor
from data.data_module import EcDataset, NonLinRegDataModule
from utils.visualise import plot_losses
from utils.utils import seed_everything, next_version_folder_name

if torch.cuda.is_available():
    dev = "cuda"
else:
    dev = "cpu"

PERCENT_VAL_EXAMPLES = 0.7
PERCENT_TRAIN_EXAMPLES = 0.7
DIR = os.getcwd()
LEAD_TIME = 4 # 2.5 days
LOOKBACK = 1 # 2 days
BATCHSIZE = 6

#db_path = "/perm/daep/ec_land_db_test/ecland_i6aj_2018_2022_6H.zarr"
db_path = "/perm/daep/ec_land_db_test/ecland_i6aj_2016_2022_europe.zarr"

def objective(trial, epochs):

    num_layers = trial.suggest_int('num_layers', 1, 4)
    hidden_size = []
    for i in range(num_layers):
        layer_name = f'n_units_layer_{i}'
        n_units = trial.suggest_int(layer_name, 4, 128)
        hidden_size.append(n_units)
    #hidden_size = [128, 256, 128]
        
    dropout = trial.suggest_float("dropout", 0.11, 0.24)
    weight_decay = trial.suggest_float("weight_decay", 0.0001, 0.002, log=True)
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 0.05, log=True)
    loss = trial.suggest_categorical('loss', ['mse', 'smoothl1'])
    
    print("Setting up data Module and load data set")
    dataset = EcDataset(start_yr = "2016",
                   end_yr= "2018",
                    x_idxs=(0, None),
                    lat_lon=None,
                   path=db_path) 

    print("Setting up data Module and load data set")
    data_module = NonLinRegDataModule(start_yr = "2016",
                                 end_yr= "2018",
                                 x_idxs=(0, None),
                                 lat_lon=None,
                                  lead_time = LEAD_TIME,
                                  lookback = LOOKBACK,
                                  batchsize = BATCHSIZE,
                                  db_path=db_path)

    # Debugging step
    print("Model architecture boundaries")
    input_dim = len(dataset.dynamic_feat_lst + dataset.static_feat_lst) # Number of input features plus cyclic features.
    target_dim = len(dataset.targ_lst)  # Number of output targets
    spatial_points = dataset.X_static.shape[0] # Number of spatial points
    print("Input dimemnsion: ", input_dim)
    print("Target dimemnsion: ", target_dim)
    print("Number of spatial points: ", spatial_points)
    
        # Set up model
    print("Set up model")
    model = MLPregressor(input_size=input_dim, hidden_size=hidden_size, output_size = target_dim, batch_size=BATCHSIZE,
                         learning_rate=learning_rate, lookback = LOOKBACK, rollout=LEAD_TIME, dropout=dropout, loss = loss,
                        weight_decay = weight_decay)

    version = next_version_folder_name('src/mlp_2D/nas/mlp_temp')
    csv_logger = CSVLogger('src/mlp_2D/nas', name='mlp_temp', version = version)  # Change 'logs' to the directory where you want to save the logs

    print("Setting up trainer.")
    # torch.set_float32_matmul_precision('medium')
    trainer = L.Trainer(
        #precision='bf16-mixed',
        logger=csv_logger,
        limit_val_batches=PERCENT_VAL_EXAMPLES,
        limit_train_batches=PERCENT_TRAIN_EXAMPLES,
        enable_checkpointing=False,
        max_epochs=epochs,  # 40  # 100,  # 200,
        gradient_clip_val=0.4,
        accelerator=dev,
        callbacks=[PyTorchLightningPruningCallback(trial, monitor="val_loss")],
    )
    
    hyperparameters = dict(batch_size = BATCHSIZE, dropout=dropout, learning_rate = learning_rate, num_layers = len(hidden_size), loss = loss,
                           lookback = LOOKBACK, lead_time = LEAD_TIME, weight_decay = weight_decay, hidden_size = hidden_size)
    trainer.logger.log_hyperparams(hyperparameters)
    
    print("Fitting the model.")
    start_time = time.time()
    trainer.fit(model, data_module)
    end_time = time.time()
    duration = (end_time - start_time)/60

    return trainer.callback_metrics["val_loss"].item()


if __name__ == "__main__":

    n_trials = 150
    max_epochs = 15
    timeout = 172700 # two days.

    pruner = optuna.pruners.NopPruner() #if args.pruning optuna.pruners.MedianPruner() else optuna.pruners.NopPruner()

    study = optuna.create_study(direction="minimize", pruner=pruner)
    study.optimize(lambda trial: objective(trial, max_epochs),
                       n_trials=n_trials, timeout = timeout)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
