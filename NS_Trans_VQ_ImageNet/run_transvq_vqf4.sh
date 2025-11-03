#!/bin/bash 
#SBATCH --job-name=vqvae_ldm         #  
#SBATCH --output=logs_transvq/vqvae_ldm_%j.out  #  
#SBATCH --error=logs_transvq/vqvae_ldm_%j.err 
#SBATCH --nodes=1                #  
#SBATCH --ntasks=1               #  
#SBATCH --cpus-per-task=4        #  
#SBATCH --gres=gpu:1             #  
#SBATCH --time=6-23:59:59        #  
#SBATCH --partition=ciaq        #  
#SBATCH --mem=16G

echo "Job started on $(date)" 
echo "Running on node: $(hostname)"  

export CUBLAS_WORKSPACE_CONFIG=:4096:8
python main.py --base configs/vqvae/vq-f4_transvq.yaml -t True --gpus 0 -l logs_transvq -n transvq_v1 --no-test true

echo "Job finished on $(date)" 