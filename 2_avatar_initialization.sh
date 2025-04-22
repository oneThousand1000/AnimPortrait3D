#!/bin/bash

# Run sample
# bash 0_assets_mesh_gen.sh face_id /path/to/trigrid /path/to/prompt /path/to/output
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
avatar_init_dir=$output_dir/avatar_init/$face_id
mesh_output_dir=$output_dir/asset_mesh/$face_id
echo -e "${Green}Start generating asset mesh for face $face_id"
echo "Trigrid path: $trigrid_path"
echo "Prompt path: $prompt_path"
echo "Pretrained model directory: $pretrained_model_dir"
echo "Output directory: $output_dir"
echo "Avatar init directory: $avatar_init_dir"

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

if [ ! -d $avatar_init_dir ]; then
    echo -e "${Blue}Create avatar init directory${NC}"
    mkdir -p $avatar_init_dir
fi

 
if [ ! -f $avatar_init_dir/raw_images/cameras.json ]; then
    echo -e "${Blue}Generate images for gs appearance optimization${NC}"
    cd ./RiggedPointCloudGen/3DPortraitGAN_pyramid/ 
    python gen_data_for_gs_optimization.py \
        --output_dir=$avatar_init_dir/raw_images  \
        --ckpt_path=$trigrid_path  \
        --resolution=512  \
        --network=./models/model_512.pkl  \
        --render_normal=False \
        --yaw_num=24  \
        --pitch_num=5

    cd ../..
fi



if [ ! -d ./RiggedPointCloudGen/MODNet ]; then
    cd ./RiggedPointCloudGen/
    echo -e "${Blue}Clone MODNet repository${NC}"
    git clone git@github.com:ZHKKKe/MODNet.git
    cd ..
fi


if [ ! -d ./RiggedPointCloudGen/MODNet ]; then
    echo -e "${Red}Clone MODNet repository failed!${NC}"
    exit 1
fi

if [ ! -f $avatar_init_dir/raw_images/0059.png ]; then
    echo -e "${Blue}Utilize MODNet to estimate foreground masks${NC}"
    cd ./RiggedPointCloudGen/
    python fg_mask_estimate_gs.py  \
        --image_dir=$avatar_init_dir/raw_images/original_images \
        --ckpt_path=$pretrained_model_dir/MODNet/modnet_webcam_portrait_matting.ckpt
    cd ..
fi

rigged_point_cloud_path=$mesh_output_dir/rigged_point_cloud.ply
if [ ! -f $rigged_point_cloud_path ]; then
    echo -e "${Blue}Generate rigged point cloud${NC}"
    cd ./RiggedPointCloudGen/mesh_optim
    python get_colored_points_cloud.py \
        --data_dir=$mesh_output_dir
    cd ../..
fi


raw_train_steps=3000
ism_data_train_steps=1000

init_gs_path=$avatar_init_dir/output/point_cloud/iteration_$raw_train_steps/point_cloud.ply
fitted_parameters_path=$mesh_output_dir/meshes/fitted_smplx/fitted_params.pkl
 
if [ ! -f $init_gs_path ]; then
    echo -e "${Blue}Train the GS Avatar using rendered images${NC}"
    cd ./GSAvatar
    python train.py \
        -s $avatar_init_dir/raw_images \
        -m $avatar_init_dir/output --save_log_image  \
        --port 60000 --eval --white_background --bind_to_mesh \
        --lambda_scale=1e4 --threshold_scale=0.2    --lambda_xyz=1e5 --threshold_xyz=1  \
        --iterations $raw_train_steps --densify_from_iter 0  --densification_interval=200 \
        --fitted_parameters=$fitted_parameters_path  \
        --points_cloud_path=$rigged_point_cloud_path
    cd ..
fi

init_vis_path=$avatar_init_dir/vis.mp4

if [ ! -f $init_vis_path ]; then
    echo -e "${Green}Render test video, please check the output video at $init_vis_path${NC}"
    cd ./GSAvatar
    python render_animation.py \
        --points_cloud=$init_gs_path  \
        --fitted_parameters=$fitted_parameters_path  \
        --video_out_path=$init_vis_path  \
        --exp_path=./test_motion/expressions.npy \
        --pose_path=./test_motion/poses.npy
    cd ..
fi