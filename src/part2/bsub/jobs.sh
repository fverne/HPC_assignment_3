#!/bin/bash

# Just some notes for scripting
# bsub -J <JOB_NAME> < script.sh 

# First comparison
# They do not work, just do it manually
# export EXECUTABLE="poisson_jomp" && bsub -J "$EXECUTABLE" < ./omp_jomp.sub
# export EXECUTABLE="poisson_jexpand" && bsub -J "$EXECUTABLE" < ./omp_jomp.sub 