#!/bin/bash
#SBATCH --job-name=thesis_job
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --time=2-00:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --cpu-freq=highm1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=spandreas@ece.auth.gr

module load gcc miniconda3
source $CONDA_PROFILE/conda.sh

#rm -rf /mnt/scratch_b/users/s/spandreas/.conda/envs/thesis
#conda create -n thesis python=3.9 -y
conda activate thesis
cd ~/thesis
ls
pip install -U -r requirements.txt
python train.py