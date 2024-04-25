#!/bin/bash

idx_start=0
NMC=10
declare -a Js=("256" "1024" "4096")

dir_name="./results_sweep_M_diag_noisy/"
mkdir -p ${dir_name}

COUNT=0
for ((idx=idx_start;idx<NMC;idx++)); do
    for J in "${Js[@]}"; do
        job_name="idx${idx}_J${J}"
        std="${dir_name}R-%x.%j"
        scommand="sbatch --job-name=${job_name} --output=${std}.out --error=${std}.err sweep_M_diag.sbatch ${idx} ${J} ${dir_name}"
        
        echo "submit command: $scommand"
        
        $scommand
        
        (( COUNT++ ))
    done
done

echo ${COUNT} jobs
