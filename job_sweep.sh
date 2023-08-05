#!/bin/bash

NMC=100
declare -a Js=("512" "2048" "8192")

COUNT=0
for ((idx=0;idx<NMC;idx++)); do
    for J in "${Js[@]}"; do
        job_name="idx${idx}_J${J}"
        std="./results/R-%x.%j"
        scommand="sbatch --job-name=${job_name} --output=${std}.out --error=${std}.err sweep_rfm.sbatch ${idx} ${J}"
        
        echo "submit command: $scommand"
        
        $scommand
        
        (( COUNT++ ))
    done
done

echo ${COUNT} jobs
