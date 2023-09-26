#!/bin/bash

#SBATCH --time=168:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --gres gpu:1
#SBATCH --export=ALL
#SBATCH --mail-user=username@domain.edu
#SBATCH --mail-type=FAIL

echo MCidx${1}_resolutionJ${2}_mypath${3}

conda activate operator

const_N="0.000003"
declare -a Ns=("10" "23" "54" "124" "288" "668" "1548")
declare -a Ms=("100" "1000" "10000")

for M in "${Ms[@]}"; do
    for N in "${Ns[@]}"; do
        lam=$(bc -l <<< "${const_N}/sqrt(${N})")
        srun python -u train.py ${M} ${N} ${2} ${1} ${lam} ${3}
    done
done

echo done
