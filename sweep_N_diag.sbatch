#!/bin/bash

#SBATCH --time=10:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --gres=gpu:p100:1
#SBATCH --partition=gpu
#SBATCH --export=ALL
#SBATCH --mail-user=nnelsen@caltech.edu
#SBATCH --mail-type=FAIL

echo MCidx${1}_resolutionJ${2}_mypath${3}

source /home/nnelsen/miniconda3/etc/profile.d/conda.sh
conda activate fno

const_N="0.000003"
declare -a Ns=("10" "23" "54" "124" "288" "668" "1548")
declare -a Ms=("254" "385" "591" "895" "1364" "2077" "3162")
#declare -a Ms=("80" "122" "187" "283" "431" "657" "1000")

for (( i = 0; i < "${#Ns[@]}"; i++ )); do
lam=$(bc -l <<< "${const_N}/sqrt(${Ns[$i]})")
srun python -u train.py ${Ms[$i]} ${Ns[$i]} ${2} ${1} ${lam} ${3}
done

echo done
