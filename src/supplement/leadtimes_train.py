import pytorch_lightning as L
import pandas as pd
import time
import os
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
from lstm_2D.models import LSTM_m2m_autoencoder_nc as LSTMregressor
from lstm_2D.ec_database import EcDataset, NonLinRegDataModule
from utils.visualise import plot_losses
from utils.utils import get_hp_search_results, best_trial, next_version_folder_name, seed_everything

if torch.cuda.is_available():
    dev = "cuda"
else:
    dev = "cpu"
DEVICE = torch.device(dev)
print(DEVICE)

# Instantiate the parser
parser = argparse.ArgumentParser(description='Train the emulator at different lead times.')
parser.add_argument('pos_arg', type=int,
                    help='Required integer to specify training lead time')
args = parser.parse_args()
print('parsed for LEADTIME:', args.pos_arg)

seed_everything(123)
print("SEED at :", 123)
continue_training = False
results = 'analyses/europe/leadtime'

use_trial = 'src/lstm_2D/results_m2m/europe/autoencoder_nc'
with open(os.path.join(use_trial, "hparams.yaml"), "r") as stream:
                hpars = yaml.safe_load(stream)
        
print('Using version:', use_trial)
print('with hpars: ', hpars)

LOOKBACK = hpars['lookback']
LEADTIME = args.pos_arg #hpars['lead_time']
BATCHSIZE = 4 #hpars['batch_size']

print('LOOKBACK:', LOOKBACK)
print('LEADTIME:', LEADTIME)
print('BATCHSIZE:', BATCHSIZE)


print("Setting up data Module and load data set")
dataset = EcDataset(start_yr = "2016",
                   end_yr= "2021",
                    x_idxs=(0, None),
                    lat_lon=None,
                   path="/perm/daep/ec_land_db_test/ecland_i6aj_2016_2022_europe.zarr") 

print("Setting up data Module and load data set")
data_module = NonLinRegDataModule(start_yr = "2016",
                                 end_yr= "2021",
                                 x_idxs=(0, None),
                                 lat_lon=None,
                                  batchsize = BATCHSIZE,
                                  rollout = LEADTIME,
                                  lookback = LOOKBACK,
                                  db_path="/perm/daep/ec_land_db_test/ecland_i6aj_2016_2022_europe.zarr")

# Debugging step
print("Model architecture boundaries")
input_dim = len(dataset.dynamic_feat_lst + dataset.static_feat_lst)  # Number of input features
target_dim = len(dataset.targ_lst)  # Number of output targets
spatial_points = dataset.X_static.shape[0] # Number of spatial points
print("Input dimemnsion: ", input_dim)
print("Target dimemnsion: ", target_dim)
print("Number of spatial points: ", spatial_points)

# Set up model
print("Set up model")
model_pyt = LSTMregressor(input_features = input_dim, 
                          lookback = LOOKBACK, 
                          lead_time = LEADTIME, 
                          spatial_points=spatial_points, 
                          batch_size=BATCHSIZE, 
                          num_layers_en = hpars['num_layers_en'], 
                          num_layers_de = hpars['num_layers_de'], 
                          hidden_size_en = hpars['hidden_size_en'], 
                          hidden_size_de = hpars['hidden_size_de'], 
                          dropout = hpars['dropout'], 
                          learning_rate = hpars['learning_rate'], 
                          weight_decay = hpars['weight_decay'],
                          loss = hpars['loss'], 
                          use_dlogits = True)

# Convert model to half precision
#model_pyt = model_pyt.half()

if continue_training:
    path_to_checkpoint = os.path.join(os.path.join(os.path.join('src/lstm_2D', results), version), 'checkpoints')
    use_checkpoint = os.listdir(path_to_checkpoint)[-1]
    path_to_best_checkpoint = os.path.join(path_to_checkpoint, use_checkpoint)  # trainer.checkpoint_callback.best_model_path
    print("Continue training model: ", path_to_best_checkpoint)
    checkpoint = torch.load(path_to_best_checkpoint)
    torch.set_float32_matmul_precision("high")
    model_pyt.load_state_dict(checkpoint['state_dict'])

version = next_version_folder_name(os.path.join('src/evaluation', results))
csv_logger = CSVLogger('src/evaluation', name=results, version = f'leadtime_{LEADTIME}')  # Change 'logs' to the directory where you want to save the logs

torch.set_float32_matmul_precision('medium')
checkpoint_callback = ModelCheckpoint(
    monitor='val_loss_total', 
    filename='best-model-{epoch:02d}-{val_loss:.2f}',
    save_top_k=1,  # Save the best model
    save_last=True,  # Also save the last model at the end of training
)


print("Setting up trainer.")
# torch.set_float32_matmul_precision('medium')
trainer = L.Trainer(
        #precision='bf16-mixed',
        logger=csv_logger,
        max_epochs=220,  # 40  # 100,  # 200,
        enable_progress_bar=True,
        callbacks=[checkpoint_callback],
        #gradient_clip_val=0.2,
        accelerator='gpu',#dev,
        strategy='ddp',
        devices=2, 
    )

print("Fitting the model.")
start_time = time.time()
trainer.fit(model_pyt, data_module)
end_time = time.time()
duration = (end_time - start_time)/60

print("Model fitted.")
print(f"The training took {duration} minute.")

#plot_losses(pd.read_csv(os.path.join(os.getcwd(),csv_logger.experiment.metrics_file_path)))
