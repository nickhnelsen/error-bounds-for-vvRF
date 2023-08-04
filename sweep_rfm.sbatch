#!/bin/bash

#SBATCH --time=99:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --gres gpu:1
#SBATCH --export=ALL
#SBATCH --mail-user=nnelsen@caltech.edu
#SBATCH --mail-type=FAIL

echo MCidx${1}_resolutionJ${2}

source /home/nnelsen/miniconda3/etc/profile.d/conda.sh
conda activate fno

declare -a Ns=("10" "23" "54" "124" "288" "668" "1548")
declare -a Ms=("10" "32" "100" "316" "1000" "3162" "10000")

for N in "${Ns[@]}"
do
for M in "${Ms[@]}"
do
    srun python -u run_sweep_script.py $M $N ${2} ${1} | tee result_idx${1}_J${2}_m${M}_n${N}.out
done
done
echo done


-------------OLD--------------------
echo ${1:0:1}_n${3}_d${4}_${5}_L${6}_m${7}_w${8}_md${9}_wd${10}_${2::-7}

source /home/nnelsen/miniconda3/etc/profile.d/conda.sh
conda activate fno

srun python -u train_fno.py ${1} ${2} ${3} ${4} ${5} ${6} ${7} ${8} ${9} ${10}

echo done
