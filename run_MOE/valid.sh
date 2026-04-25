#!/bin/bash
# ========= Part 1 : Job Parameters ============
#SBATCH --partition=gpu
#SBATCH --qos=blnormal
#SBATCH --account=bldesign
#SBATCH --ntasks=2
#SBATCH --mem-per-cpu=12G
#SBATCH --gres=gpu:1
#SBATCH --job-name=NippMOE1
#========== Part 2 : Job workload =============
export PATH=$PATH:/scratchfs/heps/zhanf/parallel_fdmnes
export PATH=/ihepfs/bldesign/user/zhanf/anaconda3/bin:$PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/ihepfs/bldesign/user/zhanf/anaconda3/lib
source /ihepfs/bldesign/user/zhanf/anaconda3/bin/activate
export MKL_THREADING_LAYER=GNU 
cd /ihepfs/bldesign/user/zhanf/pyg_1
python valid_moe.py  99 > log0.txt 2>&1 &
sleep 60
timeout -k 2m nvidia-smi -l 10 --query-gpu=timestamp,name,index,utilization.gpu,memory.total,memory.used,power.draw --format=csv --filename=nvidia0report.csv 
wait
