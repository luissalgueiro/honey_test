#!/bin/bash
#SBATCH --job-name=Honey_SW
#SBATCH -p gpi.compute
#SBATCH --gres=gpu:1
#SBATCH --output=/home/usuaris/imatge/lsalgueiro/git/honey/src/logs/train_%j.out
#SBATCH --error=/home/usuaris/imatge/lsalgueiro/git/honey/src/logs/train_%j.err
#SBATCH --time=23:59:00
#SBATCH --mem=20G
#SBATCH -c 4
#SBATCH -w gpic12

module unload cuda
module load cuda/11.3
source activate pl_env_cu11

wandb agent salgueiro/HoneySweep/ire65n3h