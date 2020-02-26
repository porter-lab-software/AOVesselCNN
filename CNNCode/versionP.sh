#!/bin/bash
#SBATCH -J versionPrint       # job name
#SBATCH -o versionPrint.o%j       # output and error file name (%j expands to jobID)
#SBATCH -N 1 -n 2 
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -A porter
#SBATCH --mail-user=hqueener@central.uh.edu
#SBATCH --mail-type=all  # email me when the job starts and ends
#SBATCH --mem=100GB
#SBATCH -t 24:00:00

#SBATCH --mail-type=all  # email me when the job starts


module load Anaconda2/python-2.7  
module add CUDA/9.1.85
python  version_print.py
nvcc --version