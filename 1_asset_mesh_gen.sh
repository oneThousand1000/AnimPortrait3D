#!/bin/bash

RED='\033[0;31m'
Green='\033[0;32m'
Blue='\033[0;34m'
NC='\033[0m'



# Read the input arguments
face_id=$1
input_dir=$2
trigrid_path="${input_dir}/${face_id}.pth"
prompt_path="${input_dir}/${face_id}.txt"
pretrained_model_dir=$3
output_dir=$4
mesh_output_dir=$output_dir/asset_mesh/$face_id

echo -e "${Green}Start generating asset mesh for face $face_id"
echo "Trigrid path: $trigrid_path"
echo "Prompt path: $prompt_path"
echo "Pretrained model directory: $pretrained_model_dir"
echo "Output directory: $output_dir"
echo "Mesh output directory: $mesh_output_dir"

# Check if the input paths are existing
if [ ! -f $trigrid_path ]; then
    echo -e "${Red}The trigrid path does not exist.${NC}"
    exit 1
fi

if [ ! -f $prompt_path ]; then
    echo -e "${Red}The prompt path does not exist.${NC}"
    exit 1
fi

# Create the output directory if it does not exist
if [ ! -d $output_dir ]; then
    echo -e "${Blue}Create output directory${NC}"
    mkdir -p $output_dir
fi


if [ ! -d $mesh_output_dir ]; then
    echo -e "${Blue}Create mesh output directory${NC}"
    mkdir -p $mesh_output_dir
fi
 
 
 
if [ ! -d $mesh_output_dir/face_segment ]; then
    echo -e "${Red}Please refer to the readme to run sapiens to generate face segmentation first!${NC}"
    exit 1
fi

if [ ! -d $mesh_output_dir/hair_segment ]; then
    echo -e "${Blue}Generate hair segmentation${NC}"
    cd ./RiggedPointCloudGen/pytorch-hair-segmentation
    python  test.py  \
        --networks pspnet_resnet101 \
        --ckpt_dir ./pspnet_resnet101_sgd_lr_0.002_epoch_100_test_iou_0.918.pth \
        --dataset MyDataset \
        --data_dir $mesh_output_dir/images \
        --save_dir $mesh_output_dir/hair_segment \
        --use_gpu True  
    cd ../..
fi

if [ ! -f $mesh_output_dir/mesh_segment_log/hair.obj ]; then
    echo -e "${Blue}Segment mesh into hair and face${NC}"
    cd ./RiggedPointCloudGen/mesh_optim
    python segment_mesh.py \
        --data_dir=$mesh_output_dir
    cd ../..
fi

if [ ! -f $mesh_output_dir/mesh_fit_log/flame_fitting/fitted_flame_mesh_0.obj ]; then
    echo -e "${Blue}Fitting flame model${NC}"
    cd ./RiggedPointCloudGen/mesh_optim/flame_fitting
    python vhap/track_single_frame.py \
        --data.root_folder "${mesh_output_dir}"  \
        --exp.output_folder $mesh_output_dir/mesh_fit_log/flame_fitting  \
        --data.sequence="${face_id}" 
    cd ../../..
fi

if [ ! -f $mesh_output_dir/meshes/composed_smplx_portrait3d/fitted_params.pkl ]; then
    echo -e "${Blue}Fitting smplx model${NC}"
    cd ./RiggedPointCloudGen/mesh_optim
    python smplx_fit.py  \
        --data_dir=$mesh_output_dir
    cd ../..
fi