import pytorch_lightning as L
import torch
import torch.nn as nn
import yaml
import csv
import pandas as pd
import time
import sys
import os
import argparse

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
print(SCRIPT_DIR)

from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from mlp_2D.models import MLPregressor_global as MLPregressor
from data.data_module import EcDataset, NonLinRegDataModule
from utils.visualise import plot_losses
from utils.utils import get_hp_search_results, best_trial, next_version_folder_name

parser = argparse.ArgumentParser(description="Load different config files for training.")
parser.add_argument('config_file', type=str, help='Specify .yaml file from same directory.')

args = parser.parse_args()

experiment_path = 'src/mlp'
configs_path = 'configs'

with open(os.path.join(configs_path, args.config_file)) as stream:
    try:
        CONFIG = yaml.safe_load(stream)
        print(f"Opening {args.config_file} for experiment configuration.")
    except yaml.YAMLError as exc:
        print(exc)

nas_path = os.path.join(experiment_path, 'nas/mlp')
hp_results = get_hp_search_results(nas_path, column_name='val_loss')
version = best_trial(hp_results, 'val_loss')

use_trial = os.path.join(nas_path, version)
print("Loading hpars from:", use_trial)
with open(os.path.join(use_trial, "hparams.yaml"), "r") as stream:
            HPARS = yaml.safe_load(stream)
print("Hpars:", HPARS)
    
if __name__ == "__main__":

    # Set device
    if torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"
    device = torch.device(dev)

    data_module = NonLinRegDataModule(CONFIG)
    dataset = EcDataset(CONFIG,
                        CONFIG["validation_start"],
                        CONFIG["validation_end"])
    
    if CONFIG["logging"]["logger"] == "csv":
        logger = CSVLogger(
            CONFIG["logging"]["location"], name=CONFIG["logging"]["name"]
        )  # Change 'logs' to the directory where you want to save the logs
    elif CONFIG["logging"]["logger"] == "mlflow":
        logger = MLFlowLogger(
            experiment_name=CONFIG["logging"]["project"],
            run_name=CONFIG["logging"]["name"],
            tracking_uri=CONFIG["logging"]["uri"],  # "file:./mlruns",
        )
    else:
        logger = None

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss', 
        filename='best-model-{epoch:02d}-{val_loss:.2f}',
        save_top_k=1,  # Save the best model
        save_last=True,  # Also save the last model at the end of training
    )

    # train
    #logging.info("Setting model params...")
    input_clim_dim = len(dataset.static_feat_lst)
    input_met_dim = len(dataset.dynamic_feat_lst)
    input_state_dim = len(dataset.targ_lst)
    output_dim = len(dataset.targ_lst)  # Number of output targets
    output_diag_dim = 0 # len(dataset.targ_diag_lst)
    

    model = MLPregressor(input_clim_dim = input_clim_dim,
                             input_met_dim = input_met_dim,
                             input_state_dim = input_state_dim,
                             hidden_dim = HPARS['hidden_size'],
                             output_dim = output_dim,
                             output_diag_dim = output_diag_dim,
                             batch_size=CONFIG["batch_size"],
                             learning_rate=HPARS['learning_rate'], 
                             lookback = None, 
                             rollout=CONFIG["roll_out"], 
                             dropout=HPARS['dropout'], 
                             weight_decay = HPARS['weight_decay'], 
                             loss = HPARS['loss'],
                             device = device, 
                             targets = CONFIG["targets_prog"],
                             db_path = CONFIG["file_path"])
    
    torch.set_float32_matmul_precision("high")
    
    print("Setting up trainer.")
    # torch.set_float32_matmul_precision('medium')
    trainer = L.Trainer(
        precision='bf16-mixed',
        logger=logger,
        max_epochs=CONFIG["max_epochs"],  # 40  # 100,  # 200,
        enable_progress_bar=True,
        callbacks=[checkpoint_callback],
        # barebones=True,
        accelerator='gpu',#dev,
        strategy=CONFIG["strategy"],
        devices=CONFIG["devices"], 
    )
    
    if CONFIG["continue_training"]:
        path_to_checkpoint = os.path.join(os.path.join(CONFIG['logging']['location'], CONFIG['logging']['name']), 'version_0/checkpoints')
        use_checkpoint = os.listdir(path_to_checkpoint)[-1]
        path_to_best_checkpoint = os.path.join(path_to_checkpoint, use_checkpoint)  # trainer.checkpoint_callback.best_model_path
        print("Continue training model: ", path_to_best_checkpoint)
        checkpoint = torch.load(path_to_best_checkpoint)
        torch.set_float32_matmul_precision("high")
        model.load_state_dict(checkpoint['state_dict'])

    start_time = time.time()
    print("Fitting the model.")
    trainer.logger.log_hyperparams(CONFIG)
    trainer.fit(model, data_module)
    end_time = time.time()
    duration = (end_time - start_time)/60
    print("Time required for fitting ... minutes ...:", duration)
    
    #model_pyt.eval()
    #torch.save(model_pyt.state_dict(), CONFIG["model_path"])





