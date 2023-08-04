#!/bin/bash

#SBATCH --time=99:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --gres gpu:1
#SBATCH --mem=32G
#SBATCH --job-name="sweep"
#SBATCH --output=R-%x.%j.out
#SBATCH --error=R-%x.%j.err
#SBATCH --export=ALL
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

declare -a Js=("512" "2048" "8192")
declare -a Ns=("10" "23" "54" "124" "288" "668" "1548")
declare -a Ms=("10" "32" "100" "316" "1000" "3162" "10000")

conda activate operator

for J in "${Js[@]}"
do
for N in "${Ns[@]}"
do
for M in "${Ms[@]}"
do
    srun python -u run_sweep_script.py $M $N $J | tee result_J${J}_m${M}_n${N}.out
done
done
done
echo done
