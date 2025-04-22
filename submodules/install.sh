#!/bin/bash

git clone https://github.com/skhu101/GauHuman
cp -r GauHuman/submodules/diff-gaussian-rasterization/ ./diff-gaussian-rasterization
 

cd diff-gaussian-rasterization
pip install -e .
cd ..


git clone https://github.com/ShenhanQian/GaussianAvatars.git --recursive
cp -r GaussianAvatars/submodules/simple-knn . 
 

cd simple-knn
pip install -e .
cd ..


 
 