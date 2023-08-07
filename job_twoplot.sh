#!/bin/bash

NMC=10
declare -a Js=("256" "1024" "4096")

dir_name="./results/"
mkdir -p ${dir_name}

COUNT=0
for ((idx=0;idx<NMC;idx++)); do
    for J in "${Js[@]}"; do
        job_name="idx${idx}_J${J}"
        std="${dir_name}R-%x.%j"
        scommand="sbatch --job-name=${job_name} --output=${std}.out --error=${std}.err twoplot_rfm.sbatch ${idx} ${J}"
        
        echo "submit command: $scommand"
        
        $scommand
        
        (( COUNT++ ))
    done
done

echo ${COUNT} jobs
