#!/bin/bash
RED='\033[0;31m'
Green='\033[0;32m'
Blue='\033[0;34m'
NC='\033[0m'

#Install torch, torchvision, and torch-scatter 
echo -e "${Green}Install torch, torchvision, and torch-scatter${NC}"
pip install torch==2.3.0+cu121 torchvision==0.18.0+cu121  -f https://download.pytorch.org/whl/torch_stable.html
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.3.0+cu121.html


#Install other packages# 
echo -e "${Green}Install other packages${NC}"
pip install -r requirements.txt


#Install submodules#
echo -e "${Green}Install submodules${NC}"
cd submodules
bash ./install.sh
cd ..


#OSMesa Dependencies#
echo -e "${Green}Install OSMesa Dependencies${NC}"
sudo apt install  libosmesa6  libosmesa6-dev


#Install kaolin#
echo -e "${Green}Install kaolin${NC}"
pip install kaolin==0.17.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.3.0_cu121.html
 

#Install pytorch3D#
echo -e "${Green}Install pytorch3D${NC}"
pip install https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu121_pyt230/pytorch3d-0.7.6-cp310-cp310-linux_x86_64.whl


#Install a modified version of VHAP (Versatile Head Alignment with Adaptive Appearance Priors) for flame fitting#
echo -e "${Green}Install a modified version of VHAP (Versatile Head Alignment with Adaptive Appearance Priors) for flame fitting${NC}"
cd ./RiggedPointCloudGen/mesh_optim/flame_fitting
pip install -e .

cd ../../..
