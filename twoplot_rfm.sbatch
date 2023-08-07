#!/bin/bash

#SBATCH --time=99:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --gres gpu:1
#SBATCH --export=ALL
#SBATCH --mail-user=nnelsen@caltech.edu
#SBATCH --mail-type=FAIL

echo MCidx${1}_resolutionJ${2}

source /home/nnelsen/miniconda3/etc/profile.d/conda.sh
conda activate fno

Nfix=1548
Mfix=10000
declare -a Ns=("10" "23" "54" "124" "288" "668" "1548")
declare -a Ms=("10" "32" "100" "316" "1000" "3162" "10000")

for M in "${Ms[@]}"
do
    srun python -u run_sweep_script.py ${M} ${Nfix} ${2} ${1}
done

for N in "${Ns[@]}"
do
    srun python -u run_sweep_script.py ${Mfix} ${N} ${2} ${1}
done

echo done
