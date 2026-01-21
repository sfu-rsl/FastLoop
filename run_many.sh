cd Examples

pathDatasetTUM_VI=$HOME/slam/datasets/tumvi/exported/euroc/512_16
# seq=dataset-outdoors6_512
# seq=dataset-outdoors4_512 # hard vs ldlt
# seq=dataset-outdoors3_512
# seq=dataset-room2_512
# seq=dataset-room3_512 # works best
# seq=dataset-room4_512
# seq=dataset-room5_512
# seq=dataset-room6_512
# seq=dataset-outdoors1_512

# sequences=(dataset-room1_512, dataset-room2_512, dataset-room3_512, dataset-room4_512, dataset-room5_512, dataset-room6_512)

# sequences=(dataset-room1_512, dataset-room2_512, dataset-room3_512, dataset-room4_512, dataset-room5_512, dataset-room6_512)


# sequences=(dataset-room1_512 dataset-room2_512 dataset-room3_512 dataset-room4_512 dataset-room5_512 dataset-room6_512)


# sequences=("dataset-room1_512" "dataset-room2_512" "dataset-room3_512" "dataset-room4_512" "dataset-room5_512" "dataset-room6_512")

# sequences=("dataset-corridor1_512" "dataset-corridor2_512" "dataset-corridor5_512")
# sequences=("dataset-magistrale1_512" "dataset-magistrale2_512")
sequences=("dataset-outdoors1_512" "dataset-outdoors3_512" "dataset-outdoors5_512" "dataset-outdoors7_512")


# echo ./Stereo-Inertial/stereo_inertial_tum_vi ../Vocabulary/ORBvoc.txt Stereo-Inertial/TUM-VI.yaml $pathDatasetTUM_VI/"$seq"_16/mav0/cam0/data $pathDatasetTUM_VI/"$seq"_16/mav0/cam1/data Stereo-Inertial/TUM_TimeStamps/"$seq".txt Stereo-Inertial/TUM_IMU/"$seq".txt "$seq"_stereoi
for seq in ${sequences[@]};
do
    # for i in $(seq 1 5);
    # do
    #     echo "Sequence " $seq " Run " $i
    #     CUDA_MODULE_LOADING=EAGER ./Stereo-Inertial/stereo_inertial_tum_vi ../Vocabulary/ORBvoc.txt Stereo-Inertial/TUM-VI.yaml $pathDatasetTUM_VI/"$seq"_16/mav0/cam0/data $pathDatasetTUM_VI/"$seq"_16/mav0/cam1/data Stereo-Inertial/TUM_TimeStamps/"$seq".txt Stereo-Inertial/TUM_IMU/"$seq".txt "$seq"_stereoi 0 ../Results/ORB-SLAM3/"$seq"/v1 0 0
    # done
    echo "========================================================================================="
    echo "Sequence " $seq
    CUDA_MODULE_LOADING=EAGER ./Stereo-Inertial/stereo_inertial_tum_vi ../Vocabulary/ORBvoc.txt Stereo-Inertial/TUM-VI.yaml $pathDatasetTUM_VI/"$seq"_16/mav0/cam0/data $pathDatasetTUM_VI/"$seq"_16/mav0/cam1/data Stereo-Inertial/TUM_TimeStamps/"$seq".txt Stereo-Inertial/TUM_IMU/"$seq".txt "$seq"_stereoi 0 ../Results/ORB-SLAM3/"$seq"/v1 0 0
done
# ./Stereo-Inertial/stereo_inertial_tum_vi ../Vocabulary/ORBvoc.txt Stereo-Inertial/TUM-VI_far.yaml $pathDatasetTUM_VI/"$seq"_16/mav0/cam0/data $pathDatasetTUM_VI/"$seq"_16/mav0/cam1/data Stereo-Inertial/TUM_TimeStamps/"$seq".txt Stereo-Inertial/TUM_IMU/"$seq".txt "$seq"_stereoi

