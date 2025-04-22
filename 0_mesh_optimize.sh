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
 
if [ ! -f $mesh_output_dir/meshes/original_shape.ply ]; then
    echo -e "${Blue}Generate images for mesh optimization${NC}"
    cd ./RiggedPointCloudGen/3DPortraitGAN_pyramid/ 
    python gen_data_for_mesh_optimization.py \
        --output_dir=$mesh_output_dir  \
        --ckpt_path=$trigrid_path  \
        --resolution=512  \
        --grid=1x1    \
        --network=./models/model_512.pkl  \
        --render_normal=False 

    cd ../..
fi

if [ ! -f $mesh_output_dir/images/0023_detailed.png ]; then
    echo -e "${Blue}Utilize controlnet-tile to improve the quality of the rendered raw images${NC}"
    cd ./RiggedPointCloudGen/
    python controlnet_tile_rgb.py   \
        --text_prompt_path=$prompt_path  \
        --data_dir=$mesh_output_dir  \
        --controlnettile_path=$pretrained_model_dir/control_v11f1e_sd15_tile \
        --diffusion_path=$pretrained_model_dir/Realistic_Vision_V5.1_noVAE 
    cd ..

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

if [ ! -f $mesh_output_dir/images/0023_rgba.png ]; then
    echo -e "${Blue}Utilize MODNet to estimate foreground masks${NC}"
    cd ./RiggedPointCloudGen/
    python fg_mask_estimate.py  \
        --image_dir=$mesh_output_dir/images \
        --ckpt_path=$pretrained_model_dir/MODNet/modnet_webcam_portrait_matting.ckpt
    cd ..
fi

 
 

if [ ! -f $mesh_output_dir/images/0011_predicted_normal.png ]; then
    echo -e "${Blue}Utilize Unique3D to estimate normal maps${NC}"
    cd ./RiggedPointCloudGen/Unique3D
    python predict_normals.py  \
        --data_dir=$mesh_output_dir  \
        --unique3d_model_path=$pretrained_model_dir/unique3d/ckpt
    cd ../..
fi


 

if [ ! -f $mesh_output_dir/meshes/optimized_rgba_textured_mesh/mesh.obj ]; then
    echo -e "${Blue}Optimize the mesh using the predicted normal maps${NC}"
    cd ./RiggedPointCloudGen/mesh_optim
    python train_mesh.py  \
        --data_dir=$mesh_output_dir  
    cd ../..
fi

 
 
