#!/bin/bash
RED='\033[0;31m'
Green='\033[0;32m'
Blue='\033[0;34m'
NC='\033[0m'

pretrained_model_dir=$1
project_root=$2

echo -e "${Green}Start fetching data${NC}"
echo -e "${Green}Pretrained model save directory: $pretrained_model_dir${NC}"
echo -e "${Green}Project root: $project_root${NC}"

# Download examplar Portrait3D data
echo -e "${Blue}\nDownload examplar Portrait3D data${NC}"
mkdir -p $project_root/outputs/Portrait3D_results
if [ ! -f $project_root/outputs/Portrait3D_results/000.png ]; then
    echo -e "${Blue}Downloading to $project_root/outputs/Portrait3D_results/000.png${NC}"
    gdown 18sUbspP52Flpj5XR8R0Jtwb2L9xZcvft -O $project_root/outputs/Portrait3D_results/000.png
    # Or download from 'https://drive.google.com/file/d/18sUbspP52Flpj5XR8R0Jtwb2L9xZcvft/view' and save as 'outputs/Portrait3D_results/000.png'
fi
if [ ! -f $project_root/outputs/Portrait3D_results/000.pth ]; then
    echo -e "${Blue}Downloading to $project_root/outputs/Portrait3D_results/000.pth${NC}"
    gdown 1iWP2tTJC6BQHJ71i1XiS5cQRwUFrJIpG -O $project_root/outputs/Portrait3D_results/000.pth
    # Or download from 'https://drive.google.com/file/d/1iWP2tTJC6BQHJ71i1XiS5cQRwUFrJIpG/view' and save as 'outputs/Portrait3D_results/000.pth'
fi
if [ ! -f $project_root/outputs/Portrait3D_results/000.txt ]; then
    echo -e "${Blue}Downloading to $project_root/outputs/Portrait3D_results/000.txt${NC}"
    gdown 1KtsJB4Niem1uCUGLzUQcbsuBrnv8nbAK -O $project_root/outputs/Portrait3D_results/000.txt
    # Or download from 'https://drive.google.com/file/d/1KtsJB4Niem1uCUGLzUQcbsuBrnv8nbAK/view' and save as 'outputs/Portrait3D_results/000.txt'
fi 


# Download 3DPortraitGAN files
mkdir -p $project_root/RiggedPointCloudGen/3DPortraitGAN_pyramid/models
if [ ! -f $project_root/RiggedPointCloudGen/3DPortraitGAN_pyramid/models/model_512.pkl ]; then
    echo -e "${Blue}Downloading 3DPortraitGAN model to $project_root/RiggedPointCloudGen/3DPortraitGAN_pyramid/models/model_512.pkl${NC}"
    gdown 1P6k4UwGGNmxa6-rQr2oyIOmAPiLAd_WE -O $project_root/RiggedPointCloudGen/3DPortraitGAN_pyramid/models/model_512.pkl
    # Or download from 'https://drive.google.com/file/d/1P6k4UwGGNmxa6-rQr2oyIOmAPiLAd_WE/view' and save as 'RiggedPointCloudGen/3DPortraitGAN_pyramid/models/model_512.pkl'
fi 



# Download smpl model 
urle () { [[ "${1}" ]] || return 1; local LANG=C i x; for (( i = 0; i < ${#1}; i++ )); do x="${1:i:1}"; [[ "${x}" == [a-zA-Z0-9.~-] ]] && echo -n "${x}" || printf '%%%02X' "'${x}"; done; echo; }

mkdir -p $project_root/RiggedPointCloudGen/3DPortraitGAN_pyramid/smplx_models/smpl
if [ ! -f  $project_root/RiggedPointCloudGen/3DPortraitGAN_pyramid/smplx_models/smpl/SMPL_FEMALE.pkl ]; then 
    echo -e "${Green}\nYou need to register at https://smpl.is.tue.mpg.de${NC}"
    read -p "Username (SMPL):" username
    read -p "Password (SMPL):" password
    username=$(urle $username)
    password=$(urle $password) 
    wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=smpl&sfile=SMPL_python_v.1.0.0.zip&resume=1' -O "$project_root/RiggedPointCloudGen/3DPortraitGAN_pyramid/smplx_models/smpl/SMPL_python_v.1.0.0.zip" --no-check-certificate --continue 
fi 


if [ ! -f  $project_root/RiggedPointCloudGen/3DPortraitGAN_pyramid/smplx_models/smpl/SMPL_FEMALE.pkl ]; then
    echo -e "${Green}You need to register at https://smplify.is.tue.mpg.de${NC}"
    read -p "Username (SMPLify):" username
    read -p "Password (SMPLify):" password
    username=$(urle $username)
    password=$(urle $password)
    wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=smplify&sfile=mpips_smplify_public_v2.zip&resume=1' -O "$project_root/RiggedPointCloudGen/3DPortraitGAN_pyramid/smplx_models/smpl/mpips_smplify_public_v2.zip" --no-check-certificate --continue
fi 


if [ ! -f $project_root/RiggedPointCloudGen/3DPortraitGAN_pyramid/smplx_models/smpl/SMPL_FEMALE.pkl ]; then 
    cd $project_root/RiggedPointCloudGen/3DPortraitGAN_pyramid/smplx_models/smpl
    unzip $project_root/RiggedPointCloudGen/3DPortraitGAN_pyramid/smplx_models/smpl/SMPL_python_v.1.0.0.zip
    unzip $project_root/RiggedPointCloudGen/3DPortraitGAN_pyramid/smplx_models/smpl/mpips_smplify_public_v2.zip

    mv $project_root/RiggedPointCloudGen/3DPortraitGAN_pyramid/smplx_models/smpl/smpl/models/basicModel_f_lbs_10_207_0_v1.0.0.pkl $project_root/RiggedPointCloudGen/3DPortraitGAN_pyramid/smplx_models/smpl/SMPL_FEMALE.pkl
    mv $project_root/RiggedPointCloudGen/3DPortraitGAN_pyramid/smplx_models/smpl/smpl/models/basicmodel_m_lbs_10_207_0_v1.0.0.pkl $project_root/RiggedPointCloudGen/3DPortraitGAN_pyramid/smplx_models/smpl/SMPL_MALE.pkl
    mv $project_root/RiggedPointCloudGen/3DPortraitGAN_pyramid/smplx_models/smpl/smplify_public/code/models/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl $project_root/RiggedPointCloudGen/3DPortraitGAN_pyramid/smplx_models/smpl/SMPL_NEUTRAL.pkl

    rm -r $project_root/RiggedPointCloudGen/3DPortraitGAN_pyramid/smplx_models/smpl/smpl
    rm -r $project_root/RiggedPointCloudGen/3DPortraitGAN_pyramid/smplx_models/smpl/smplify_public
    rm $project_root/RiggedPointCloudGen/3DPortraitGAN_pyramid/smplx_models/smpl/SMPL_python_v.1.0.0.zip
    rm $project_root/RiggedPointCloudGen/3DPortraitGAN_pyramid/smplx_models/smpl/mpips_smplify_public_v2.zip
 
fi


if [ ! -f $project_root/RiggedPointCloudGen/mesh_optim/flame_fitting/asset/flame/FLAME_masks.pkl ]; then
    cd $project_root/RiggedPointCloudGen/mesh_optim/flame_fitting/asset/flame
    echo -e "${Green}You need to register at https://flame.is.tue.mpg.de${NC}"
    read -p "Username (Flame):" username
    read -p "Password (Flame):" password
    username=$(urle $username)
    password=$(urle $password)

    wget 'https://files.is.tue.mpg.de/tbolkart/FLAME/FLAME_masks.zip' -O "$project_root/RiggedPointCloudGen/mesh_optim/flame_fitting/asset/flame/FLAME_masks.zip" --no-check-certificate --continue

    unzip $project_root/RiggedPointCloudGen/mesh_optim/flame_fitting/asset/flame/FLAME_masks.zip
    rm $project_root/RiggedPointCloudGen/mesh_optim/flame_fitting/asset/flame/FLAME_masks.zip
    rm $project_root/RiggedPointCloudGen/mesh_optim/flame_fitting/asset/flame/FLAME_masks.gif
    rm $project_root/RiggedPointCloudGen/mesh_optim/flame_fitting/asset/flame/readme

    

    wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=flame&resume=1&sfile=FLAME2020.zip' -O "$project_root/RiggedPointCloudGen/mesh_optim/flame_fitting/asset/flame/FLAME2020.zip" --no-check-certificate --continue
    unzip $project_root/RiggedPointCloudGen/mesh_optim/flame_fitting/asset/flame/FLAME2020.zip 


fi



if [ ! -f $project_root/RiggedPointCloudGen/mesh_optim/smplx_model/assets/SMPLX_NEUTRAL_2020.npz ]; then

    cp $project_root/RiggedPointCloudGen/mesh_optim/flame_fitting/asset/flame/FLAME_masks.pkl  $project_root/RiggedPointCloudGen/mesh_optim/smplx_model/assets/FLAME_masks.pkl
    cp $project_root/RiggedPointCloudGen/mesh_optim/flame_fitting/asset/flame/FLAME_masks.pkl  $project_root/GSAvatar/smplx_model/assets/FLAME_masks.pkl 
    
 
    echo -e "${Green}You need to register at https://smplx.is.tue.mpg.de${NC}"
    read -p "Username (SMPLX):" username
    read -p "Password (SMPLX):" password
    username=$(urle $username)
    password=$(urle $password)

    wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=smplx&sfile=SMPLX_NEUTRAL_2020.npz&resume=1' -O "$project_root/RiggedPointCloudGen/mesh_optim/smplx_model/assets/SMPLX_NEUTRAL_2020.npz" --no-check-certificate --continue
    

    wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=smplx&sfile=smplx_mano_flame_correspondences.zip&resume=1' -O "$project_root/RiggedPointCloudGen/mesh_optim/smplx_model/assets/smplx_mano_flame_correspondences.zip"

    unzip $project_root/RiggedPointCloudGen/mesh_optim/smplx_model/assets/smplx_mano_flame_correspondences.zip -d $project_root/RiggedPointCloudGen/mesh_optim/smplx_model/assets/
    rm $project_root/RiggedPointCloudGen/mesh_optim/smplx_model/assets/smplx_mano_flame_correspondences.zip
    rm $project_root/RiggedPointCloudGen/mesh_optim/smplx_model/assets/MANO_SMPLX_vertex_ids.pkl

 
    cp $project_root/RiggedPointCloudGen/mesh_optim/smplx_model/assets/SMPL-X__FLAME_vertex_ids.npy $project_root/GSAvatar/smplx_model/assets/SMPL-X__FLAME_vertex_ids.npy
    cp $project_root/RiggedPointCloudGen/mesh_optim/smplx_model/assets/SMPLX_NEUTRAL_2020.npz $project_root/GSAvatar/smplx_model/assets/SMPLX_NEUTRAL_2020.npz


fi



# Download diffusion model

if [ ! -d $pretrained_model_dir/Realistic_Vision_V5.1_noVAE ]; then
    echo -e "${Blue}Downloading SG161222/Realistic_Vision_V5.1_noVAE to $pretrained_model_dir/Realistic_Vision_V5.1_noVAE${NC}"
    huggingface-cli download SG161222/Realistic_Vision_V5.1_noVAE  --local-dir $pretrained_model_dir/Realistic_Vision_V5.1_noVAE
fi


# Download controltile
if [ ! -d $pretrained_model_dir/control_v11f1e_sd15_tile ]; then
    echo -e "${Blue}Downloading lllyasviel/control_v11f1e_sd15_tile to $pretrained_model_dir/control_v11f1e_sd15_tile${NC}"
    huggingface-cli download lllyasviel/control_v11f1e_sd15_tile  --local-dir $pretrained_model_dir/control_v11f1e_sd15_tile
fi
 

# Download AnimPortrait3D_controlnet 
if [ ! -d $pretrained_model_dir/AnimPortrait3D_controlnet ]; then
    echo -e "${Blue}Downloading onethousand/AnimPortrait3D_controlnet to $pretrained_model_dir/AnimPortrait3D_controlnet${NC}"
    huggingface-cli download onethousand/AnimPortrait3D_controlnet  --local-dir $pretrained_model_dir/AnimPortrait3D_controlnet
fi

if [ ! -d $pretrained_model_dir/sd-vae-ft-ema ]; then
    echo -e "${Blue}Downloading stabilityai/sd-vae-ft-ema to $pretrained_model_dir/sd-vae-ft-ema${NC}"
    huggingface-cli download stabilityai/sd-vae-ft-ema  --local-dir $pretrained_model_dir/sd-vae-ft-ema
fi

# Download modnet_webcam_portrait_matting
mkdir -p $pretrained_model_dir/MODNet
if [ ! -f $pretrained_model_dir/MODNet/modnet_webcam_portrait_matting.ckpt ]; then
    # https://drive.google.com/file/d/1Nf1ZxeJZJL8Qx9KadcYYyEmmlKhTADxX/view?usp=drive_link
    echo -e "${Blue}Downloading MODNet to $pretrained_model_dir/MODNet${NC}"
    gdown 1Nf1ZxeJZJL8Qx9KadcYYyEmmlKhTADxX -O $pretrained_model_dir/MODNet/modnet_webcam_portrait_matting.ckpt
fi

# Download Wuvin/Unique3D
if [ ! -d $pretrained_model_dir/unique3d/ckpt/image2normal ]; then
    echo -e "${Blue}Downloading unique3d to $pretrained_model_dir/unique3d${NC}"
    huggingface-cli download Wuvin/Unique3D  --local-dir $pretrained_model_dir/tmp  --repo-type space
    
    # remove any file except ckpt/image2normal
    mkdir -p $pretrained_model_dir/unique3d/ckpt/image2normal
    mv $pretrained_model_dir/tmp/ckpt/image2normal $pretrained_model_dir/unique3d/ckpt

    mkdir -p $project_root/RiggedPointCloudGen/Unique3D/ckpt 
    mv $pretrained_model_dir/tmp/ckpt/realesrgan-x4.onnx  $project_root/RiggedPointCloudGen/Unique3D/ckpt 

    rm -r $pretrained_model_dir/tmp
fi

# Download pspnet_resnet101_sgd_lr_0.002_epoch_100_test_iou_0.918.pth 
if [ ! -f $project_root/RiggedPointCloudGen/pytorch-hair-segmentation/pspnet_resnet101_sgd_lr_0.002_epoch_100_test_iou_0.918.pth ]; then
    echo -e "${Blue}Downloading pytorch-hair-segmentatio to $project_root/RiggedPointCloudGen/pytorch-hair-segmentation/pspnet_resnet101_sgd_lr_0.002_epoch_100_test_iou_0.918.pth${NC}"
    gdown 1w7oMuxckqEClImjLFTH7xBCpm1wg7Eg4 -O $project_root/RiggedPointCloudGen/pytorch-hair-segmentation/pspnet_resnet101_sgd_lr_0.002_epoch_100_test_iou_0.918.pth
fi


# Download sapiens
mkdir -p $pretrained_model_dir/sapiens/pretrain/checkpoints
mkdir -p $pretrained_model_dir/sapiens/seg/checkpoints
if [ ! -f $pretrained_model_dir/sapiens/pretrain/checkpoints/sapiens_1b/sapiens_1b_epoch_173_clean.pth ]; then
    echo -e "${Blue}Downloading to pretrained sapiens to $pretrained_model_dir/sapiens/pretrain/checkpoints/sapiens_1b${NC}"
    huggingface-cli download facebook/sapiens-pretrain-1b  --local-dir $pretrained_model_dir/sapiens/pretrain/checkpoints/sapiens_1b 
    
    # remove any file except ckpt/image2normal
    echo -e "${Blue}Downloading seg sapiens to $pretrained_model_dir/sapiens/seg/checkpoints/sapiens_1b${NC}"
    huggingface-cli download facebook/sapiens-seg-1b  --local-dir $pretrained_model_dir/sapiens/seg/checkpoints/sapiens_1b  
fi
