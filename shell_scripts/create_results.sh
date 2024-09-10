#!/bin/bash -x

#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256Gb
#SBATCH --time=04:00:00
#SBATCH --output=run_forecast_europe.%j
#SBATCH --exclude=ac6-300

# debugging flags (optional)
# export NCCL_DEBUG=INFO
# export PYTHONFAULTHANDLER=1

# on your cluster you might need these:
# set the network interface
# export NCCL_SOCKET_IFNAME=ib0,lo

# generic settings
WORKDIR=/home/${USER}/PycharmProjects/land-surface-emulation

cd $WORKDIR
module purge
module load conda
conda activate /perm/pamw/venvs/mytorchcuda

srun python3 src/evaluation/create_results.py --config_file_mlp mlp_europe.yaml --config_file_lstm lstm_europe.yaml --config_file_xgb xgb_europe.yaml 

