import pytorch_lightning as L
import pandas as pd
import time
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
import torch
import csv
import yaml
import sys
import os.path
import optuna
import argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from optuna.integration import PyTorchLightningPruningCallback
from lstm_2D.models import LSTM_m2m_global as LSTMregressor
from data.data_module import EcDataset, NonLinRegDataModule
from utils.visualise import plot_losses
from utils.utils import seed_everything, next_version_folder_name

parser = argparse.ArgumentParser(description="Load different config files for training.")
parser.add_argument('config_file', type=str, help='Specify .yaml file from same directory.')

args = parser.parse_args()

if torch.cuda.is_available():
    dev = "cuda"
else:
    dev = "cpu"

PERCENT_VAL_EXAMPLES = 0.7
PERCENT_TRAIN_EXAMPLES = 0.7
DIR = os.getcwd()

configs_path = 'configs'

with open(os.path.join(configs_path, args.config_file)) as stream:
    try:
        CONFIG = yaml.safe_load(stream)
        print(f"Opening {args.config_file} for experiment configuration.")
    except yaml.YAMLError as exc:
        print(exc)
        
def objective(trial, epochs):

    num_layers_en = trial.suggest_int("num_layers_en", 1, 4, log=False)
    num_layers_de = num_layers_en # trial.suggest_int("num_layers_de", 2, 4, log=False)
    lookback = trial.suggest_int("lookback", 4, 24, log=False)
    #lead_time = trial.suggest_int("lead_time", 4, 14, log=False)
    dropout = trial.suggest_float("dropout", 0.11, 0.21)
    weight_decay = trial.suggest_float("weight_decay", 0.0001, 0.002, log=True)
    learning_rate = trial.suggest_float('learning_rate', 0.00001, 0.005, log=True)
    batch_size = trial.suggest_categorical('batch_size', [8, 16, 32, 64])
    hidden_size_en = trial.suggest_categorical('hidden_size_en', [92, 128, 196, 256])
    hidden_size_de = hidden_size_en # trial.suggest_int('hidden_size', 70, 120, step=10)
    loss =  'smoothl1' #trial.suggest_categorical('loss', ['smoothl1'])
    #learn_cell = trial.suggest_categorical('learn_cell', [True, False])

    print(lookback)
    CONFIG["lookback"] = lookback
    CONFIG["batch_size"] = batch_size
    print(CONFIG["lookback"])
    data_module = NonLinRegDataModule(CONFIG)
    dataset = EcDataset(CONFIG,
                        CONFIG["validation_start"],
                        CONFIG["validation_end"])

    input_clim_dim = len(dataset.static_feat_lst)
    input_met_dim = len(dataset.dynamic_feat_lst)
    input_state_dim = len(dataset.targ_lst)
    output_dim = len(dataset.targ_lst)  # Number of output targets
    output_diag_dim = 0 # len(dataset.targ_diag_lst)
    spatial_points = dataset.x_size 
    print("Input dimemnsion: ", input_clim_dim)
    print("Target dimemnsion: ", input_state_dim)
    print("Number of spatial points: ", spatial_points)
    
        # Set up model
    print("Set up model")
    model = LSTMregressor(input_clim_dim = input_clim_dim,
                              input_met_dim = input_met_dim,
                              input_state_dim = input_state_dim,
                              lookback = lookback, 
                              lead_time = CONFIG["roll_out"], 
                              spatial_points = spatial_points, 
                              device = dev, 
                              batch_size=batch_size,
                              learning_rate=learning_rate,
                              num_layers_en = num_layers_en, 
                              num_layers_de = num_layers_de, 
                              hidden_size_en = hidden_size_en, #HPARS['hidden_size_en'], 
                              hidden_size_de = hidden_size_de, #HPARS['hidden_size_de'], 
                              dropout = dropout, 
                              weight_decay = weight_decay,
                              loss = loss, 
                              use_dlogits = True,
                              transform = CONFIG["prog_transform"], # prognostic transform informs model
                              db_path = CONFIG["file_path"])

    version = next_version_folder_name('src/lstm_2D/nas/m2m_europe')
    csv_logger = CSVLogger('src/lstm_2D/nas', name='m2m_europe', version = version)  # Change 'logs' to the directory where you want to save the logs

    print("Setting up trainer.")
    # torch.set_float32_matmul_precision('medium')
    trainer = L.Trainer(
        #precision='bf16-mixed',
        logger=csv_logger,
        limit_val_batches=PERCENT_VAL_EXAMPLES,
        limit_train_batches=PERCENT_TRAIN_EXAMPLES,
        enable_checkpointing=False,
        max_epochs=epochs,  # 40  # 100,  # 200,
        #gradient_clip_val=0.5,
        accelerator=dev,
        callbacks=[PyTorchLightningPruningCallback(trial, monitor="val_loss_logit")],
    )
    
    hyperparameters = dict(batch_size = batch_size, 
                           dropout=dropout, 
                           learning_rate = learning_rate, 
                           num_layers_en = num_layers_en,
                           num_layers_de = num_layers_de,
                           loss = loss,
                           lookback = lookback, 
                           lead_time = CONFIG["roll_out"], 
                           weight_decay = weight_decay, 
                           hidden_size_en = hidden_size_en,
                           hidden_size_de = hidden_size_de)
    
    trainer.logger.log_hyperparams(hyperparameters)
    
    print("Fitting the model.")
    start_time = time.time()
    trainer.fit(model, data_module)
    end_time = time.time()
    duration = (end_time - start_time)/60
    
    torch.cuda.empty_cache()

    return trainer.callback_metrics["val_loss_logit"].item()


if __name__ == "__main__":

    n_trials = 100
    max_epochs = 25
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
