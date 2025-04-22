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
abstract_prompt=$5
avatar_init_dir=$output_dir/avatar_init/$face_id
mesh_output_dir=$output_dir/asset_mesh/$face_id
avatar_optim_dir=$output_dir/avatar_optim/$face_id

echo -e "${Green}Start generating asset mesh for face $face_id"
echo "Trigrid path: $trigrid_path"
echo "Prompt path: $prompt_path"
echo "Pretrained model directory: $pretrained_model_dir"
echo "Output directory: $output_dir"
echo "Avatar init directory: $avatar_init_dir"
echo "Avatar optimization directory: $avatar_optim_dir"
echo "Abstract prompt: $abstract_prompt"

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

if [ ! -d $avatar_optim_dir ]; then
    echo -e "${Blue}Create avatar optimization directory${NC}"
    mkdir -p $avatar_optim_dir
fi
 
 
raw_train_steps=3000
ism_data_train_steps=1000

init_gs_path=$avatar_init_dir/output/point_cloud/iteration_$raw_train_steps/point_cloud.ply
fitted_parameters_path=$mesh_output_dir/meshes/fitted_smplx/fitted_params.pkl



controlnet_path=$pretrained_model_dir/AnimPortrait3D_controlnet
diffusion_path=$pretrained_model_dir/Realistic_Vision_V5.1_noVAE
vae_path=$pretrained_model_dir/sd-vae-ft-ema
if [ ! -d $controlnet_path ]; then
    echo -e "${Red}The controlnet path $controlnet_path does not exist. Please refer fetch_data.sh to download the model.${NC}"
    exit 1
fi
if [ ! -d $diffusion_path ]; then
    echo -e "${Red}The diffusion path $diffusion_path does not exist. Please refer fetch_data.sh to download the model.${NC}"
    exit 1
fi
if [ ! -d $vae_path ]; then
    echo -e "${Red}The vae path $vae_path does not exist. Please refer fetch_data.sh to download the model.${NC}"
    exit 1
fi

eye_pretrain_dir=$avatar_optim_dir/eye_pretrain
eye_pretrain_result_path=$eye_pretrain_dir/res/point_cloud.ply

if [ ! -f $eye_pretrain_result_path ]; then
    echo -e "${Blue}Pretrain eye region${NC}"
    cd ./GSAvatar
    python train_eyes.py  \
        --prompt="$abstract_prompt"  \
        --bind_to_mesh  \
        --bg_path=$mesh_output_dir/images/background.png  \
        --controlnet_path=$controlnet_path   \
        --diffusion_path=/$diffusion_path  \
        --vae_path=$vae_path  \
        --lambda_scale=1e4 --threshold_scale=0.2  --lambda_xyz=1e-2 --threshold_xyz=1  --position_lr_init=0.00001  \
        --gaussian_train_iter=500  \
        --fitted_parameters=$fitted_parameters_path  \
        --points_cloud=$init_gs_path  \
        --output_dir=$eye_pretrain_dir 
    cd ..
fi

mouth_pretrain_dir=$avatar_optim_dir/mouth_pretrain
mouth_pretrain_result_path=$mouth_pretrain_dir/res/final/point_cloud.ply

if [ ! -f $mouth_pretrain_result_path ]; then
    echo -e "${Blue}Pretrain mouth region${NC}"
    cd ./GSAvatar
    python train_mouth.py     \
        --prompt="$abstract_prompt"    \
        --bind_to_mesh     \
        --bg_path=$mesh_output_dir/images/background.png     \
        --controlnet_path=$controlnet_path   \
        --diffusion_path=/$diffusion_path  \
        --vae_path=$vae_path  \
        --lambda_scale=1e4 --threshold_scale=0.2  --lambda_xyz=1e-2 --threshold_xyz=1  --position_lr_init=0.00001  \
        --gaussian_train_iter=500     \
        --fitted_parameters=$fitted_parameters_path    \
        --points_cloud=$eye_pretrain_result_path     \
        --output_dir=$mouth_pretrain_dir 
    cd ..
fi

full_pretrain_dir=$avatar_optim_dir/full_pretrain
full_pretrain_result_path=$full_pretrain_dir/res/final/point_cloud.ply

if [ ! -f $full_pretrain_result_path ]; then

    #read prompt from prompt_path
    full_prompt=$(cat $prompt_path)

    echo -e "${Blue}Pretrain full avatar${NC}"
    cd ./GSAvatar
    python train_all.py     \
        --prompt="$full_prompt"    \
        --abstract_prompt="$abstract_prompt"  \
        --bind_to_mesh     \
        --bg_path=$mesh_output_dir/images/background.png     \
        --controlnet_path=$controlnet_path   \
        --diffusion_path=/$diffusion_path  \
        --vae_path=$vae_path  \
        --lambda_scale=1e4 --threshold_scale=0.2  --lambda_xyz=1e-2 --threshold_xyz=1  --position_lr_init=0.00001  \
        --gaussian_train_iter=1000     \
        --fitted_parameters=$fitted_parameters_path    \
        --points_cloud=$mouth_pretrain_result_path     \
        --output_dir=$full_pretrain_dir
    cd ..
fi

echo -e "${Green}Finish generating asset mesh for face $face_id${NC}"

echo -e "${Green}The points_cloud path is $full_pretrain_result_path${NC}" 
echo -e "${Green}The fitted_parames is saved at $mesh_output_dir/meshes/fitted_smplx/fitted_params.pkl${NC}"


echo -e "${Blue}For visualization, using the following options: --points_cloud=$full_pretrain_result_path  --fitted_parameters=$fitted_parameters_path${NC}"
