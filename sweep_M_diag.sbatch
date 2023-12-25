#!/bin/bash

#SBATCH --time=24:00:00
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
declare -a Ns=("2" "5" "16" "49" "155" "490" "1548")
declare -a Ms=("100" "178" "316" "562" "1000" "1778" "3162")
#declare -a Ns=("10" "23" "54" "124" "288" "668" "1548")
#declare -a Ms=("254" "385" "591" "895" "1364" "2077" "3162")
#declare -a Ms=("80" "122" "187" "283" "431" "657" "1000")

for (( i = 0; i < "${#Ms[@]}"; i++ )); do
lam=$(bc -l <<< "${const_M}/${Ms[$i]}")
srun python -u train.py ${Ms[$i]} ${Ns[$i]} ${2} ${1} ${lam} ${3}
done

echo done
