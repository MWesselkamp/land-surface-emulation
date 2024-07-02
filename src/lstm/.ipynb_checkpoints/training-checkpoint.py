import pytorch_lightning as L
import torch
import torch.nn as nn
import yaml
import csv
import pandas as pd
import time
import sys
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
import torch
import csv
import sys
import yaml
import argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from lstm_2D.models import LSTM_m2m_global as LSTMregressor
from data.data_module import EcDataset, NonLinRegDataModule
from utils.visualise import plot_losses
from utils.utils import get_hp_search_results, best_trial, next_version_folder_name, seed_everything

parser = argparse.ArgumentParser(description="Load different config files for training.")
parser.add_argument('config_file', type=str, help='Specify .yaml file from same directory.')

args = parser.parse_args()

experiment_path = 'src/lstm_2D'
configs_path = 'configs'

with open(os.path.join(configs_path, args.config_file)) as stream:
    try:
        CONFIG = yaml.safe_load(stream)
        print(f"Opening {args.config_file} for experiment configuration.")
    except yaml.YAMLError as exc:
        print(exc)

nas_path = os.path.join(experiment_path, 'nas/m2m_statesemb')
hp_results = get_hp_search_results(nas_path, column_name='val_loss_logit')
version = best_trial(hp_results, 'val_loss_logit')

if CONFIG["continue_training"]:
    print("CONTINUE TRAINING")
    use_trial = os.path.join(os.path.join(CONFIG['logging']['location'], CONFIG['logging']['name']), 'version_0')
    print("Loading hpars from:", use_trial)
else:
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

    device = torch.device(dev) if CONFIG["device"] is None else CONFIG["device"]

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
        monitor='val_loss_logit', 
        filename='best-model-{epoch:02d}-{val_loss:.2f}',
        save_top_k=1,  # Save the best model
        save_last=True,  # Also save the last model at the end of training
    )

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
                              lookback = CONFIG["lookback"], 
                              lead_time = CONFIG["roll_out"], 
                              spatial_points = spatial_points, 
                              device = device, 
                              batch_size=CONFIG["batch_size"],
                              learning_rate=HPARS['learning_rate'],
                              num_layers_en = HPARS['num_layers_en'], 
                              num_layers_de = HPARS['num_layers_de'], 
                              hidden_size_en = 200, #HPARS['hidden_size_en'], 
                              hidden_size_de = 200, #HPARS['hidden_size_de'], 
                              dropout = HPARS['dropout'], 
                              weight_decay = HPARS['weight_decay'],
                              loss = HPARS['loss'], 
                              use_dlogits = True,
                              transform = CONFIG["prog_transform"], # prognostic transform informs model
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
        print("CONTINUE TRAINING")
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

