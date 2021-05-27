#$ -S /bin/bash

#$ -q long
#$ -l ngpus=1
#$ -l ncpus=8
#$ -l h_vmem=16G
#$ -l h_rt=72:00:00
#$ -N test train-gpu-job

source /etc/profile
module add anaconda3/wmlce
source activate $global_storage/conda_environments/pypytorch

python trainclustering.py 