#!/bin/sh

START=0
INCR=6
END=114
for SEED in $(seq $START $INCR $END); do
        echo "${SEED}"
        export SEED
	    sbatch -o mlogs/oout${SEED}.txt \
        -e mlogs/eout${SEED}.txt \
        --job-name=fabm_${SEED} \
        jobscript.sh
        #
        echo "Job submitted"
	    sleep 1 # pause to be kind to the scheduler

done
