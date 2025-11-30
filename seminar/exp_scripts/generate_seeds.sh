#!/bin/bash

# Get initial seed (default to 1337 if not provided)
INITIAL_SEED=${1:-1337}

# Generate 30 random seeds using the initial seed
RANDOM=$INITIAL_SEED

seeds=()
for i in {1..30}; do
    seeds+=($RANDOM)
done

# Print the initial seed as a comment and the array declaration that can be copy-pasted
echo "# Generated from initial seed: $INITIAL_SEED"
echo "seeds=(${seeds[@]})"
