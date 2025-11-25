#!/bin/bash
#SBATCH --time=48:00:00 # Run time
#SBATCH --nodes 1  # Number of reaquested nodes 
#SBATCH --ntasks-per-node=1
#SBATCH --mem 400G
#SBATCH -c 30
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint="A100|H100.80gb"
#SBTACH --job-name SAM_FINE_TUNE
#SBATCH --error=SAM_FINE_TUNE_error.o%j
#SBATCH --output=SAM_FINE_TUNE_output.o%j
#SBATCH --requeue
#SBATCH --mail-user=asarker@uni-osnabrueck.de

#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END 
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
##SBATCH --mail-type=ALL

#SBATCH --signal=SIGTERM@90
echo "running in shell: " "$SHELL"

export NCCL_SOCKET_IFNAME=lo

## to force NCCL to use share memory and not infiniband
##export NCCL_IB_DISABLE=1

export XLA_FLAGS="--xla_gpu_cuda_data_dir=/home/student/a/asarker/.conda/envs/thesis2/lib"
export TMPDIR='/share/klab/argha' 

## Please add any modules you want to load here, as an example we have commented out the modules
## that you may need such as cuda, cudnn, miniconda3, uncomment them if that is your use case 
## term handler the function is executed once the job gets the TERM signal

spack load miniconda3

eval "$(conda shell.bash hook)"
conda activate thesis2




srun python /share/klab/argha/SAM_mitochondria/MitoSAM-ViT/mitosam/fine_tune_clean.py





