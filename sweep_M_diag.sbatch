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

echo MCidx${1}_resolutionJ${2}_mypath${3}

conda activate fno

const_M="0.0007"
declare -a Ns=("124" "1548")
declare -a Ms=("10" "32" "100" "316" "1000" "3162" "10000")

for N in "${Ns[@]}"; do
    for M in "${Ms[@]}"; do
        lam=$(bc -l <<< "${const_M}/${M}")
        srun python -u train.py ${M} ${N} ${2} ${1} ${lam} ${3}
    done
done

echo done
