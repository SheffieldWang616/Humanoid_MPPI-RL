#!/bin/bash
# Script to run Humanoid data collection 100 times

echo "Starting data collection runs..."

for i in {1..150}
do
    echo "Run $i of 150"
    julia src/Humanoid_datacollection_v2.jl
    
    # Check if the command executed successfully
    if [ $? -eq 0 ]; then
        echo "Run $i completed successfully"
    else
        echo "Run $i failed with exit code $?"
    fi
    
    echo "---------------------------------"
done

echo "All 100 runs completed"