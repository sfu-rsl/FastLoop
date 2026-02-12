#!/bin/bash

pathDatasetEuroc=$HOME/Desktop/slam3/Datasets/EuRoC

mode=$1
kernel_status_FT=$2
kernel_status_TM=$3
dataset_name=$4   # مثلا V102

file_name="dataset-${dataset_name}_stereoi"
statsDir="../Results/${dataset_name}"

echo "Launching ${dataset_name} with Stereo-Inertial sensor"

# ARGS="../Vocabulary/ORBvoc.txt \
#     ./Stereo-Inertial/EuRoC.yaml \
#     "${pathDatasetEuroc}"/"${dataset_name}" \
#     ./Stereo-Inertial/EuRoC_TimeStamps/${dataset_name}.txt \
#     "${file_name}" \
#     "${statsDir}" \
#     ${mode} \
#     ${kernel_status_FT} \
#     ${kernel_status_TM}"
# gdb -ex "set args $ARGS" -ex "run" ./Stereo-Inertial/stereo_inertial_euroc


./Stereo-Inertial/stereo_inertial_euroc \
    ../Vocabulary/ORBvoc.txt \
    ./Stereo-Inertial/EuRoC.yaml \
    "${pathDatasetEuroc}"/"${dataset_name}" \
    ./Stereo-Inertial/EuRoC_TimeStamps/${dataset_name}.txt \
    "${file_name}" \
    "${statsDir}" \
    ${mode} \
    ${kernel_status_FT} \
    ${kernel_status_TM}

echo "------------------------------------"
echo "Evaluation of ${dataset_name} trajectory with Stereo-Inertial sensor"

python3 -W ignore ../evaluation/evaluate3.py \
    ${pathDatasetEuroc}/${dataset_name}/mav0/state_groundtruth_estimate0/data.csv \
    f_${file_name}.txt \
    --plot ${dataset_name}_stereoi.pdf \
    --verbose

# echo "Plotting data"

# python3 ../plot.py "${statsDir}"

# files=("f_dataset-${dataset_name}_stereoi.csv"
# "f_dataset-${dataset_name}_stereoi.txt"
# "f_dataset-${dataset_name}_stereoi.png"
# "kf_dataset-${dataset_name}_stereoi.txt"
# )
# destination_directory="${statsDir}/trajectory"
# mkdir -p $destination_directory
# mv "${files[@]}" "$destination_directory"