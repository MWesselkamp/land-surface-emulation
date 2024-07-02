#!/bin/bash

# Array of target values
targets=("swvl1" "swvl2" "swvl3" "stl1" "stl2" "stl3" "snowc")

# Loop through each target and submit a job
for target in "${targets[@]}"; do
  sbatch <<EOT
#!/bin/bash -x

#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=256Gb
#SBATCH --time=01:00:00
#SBATCH --output=forecast_skill_horizons_lstm.%j
#SBATCH --exclude=ac6-300

# debugging flags (optional)
# export NCCL_DEBUG=INFO
# export PYTHONFAULTHANDLER=1

# on your cluster you might need these:
# set the network interface
# export NCCL_SOCKET_IFNAME=ib0,lo

# generic settings
WORKDIR=/home/\${USER}/PycharmProjects/ECLand_DLforecasting

cd \$WORKDIR
module purge
module load conda
conda activate /perm/pamw/venvs/mytorchcuda

srun --gres=gpu:1 python3 src/evaluation/horizons_persistence.py --config_file lstm_northern_europe_config.yaml --target $target --score scaled_anom --num_cpus 12
EOT
done
