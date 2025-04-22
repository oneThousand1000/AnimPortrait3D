import os
import sys
import argparse
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from MODNet.src.models.modnet import MODNet
import glob
from tqdm import tqdm
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str,help='RGB images dir', required=True)
    parser.add_argument('--ckpt_path', type=str,help='ckpt_path', required=True)
    
    args = parser.parse_args()
    image_dir = args.image_dir
    ckpt_path = args.ckpt_path
    # ckpt_path = './modnet_webcam_portrait_matting.ckpt'

    # define hyper-parameters
    ref_size = 512

    # define image to tensor transform
    im_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )

    # create MODNet and load the pre-trained ckpt
    modnet = MODNet(backbone_pretrained=False)
    modnet = nn.DataParallel(modnet)

    if torch.cuda.is_available():
        modnet = modnet.cuda()
        weights = torch.load(ckpt_path)
    else:
        weights = torch.load(ckpt_path, map_location=torch.device('cpu'))
    modnet.load_state_dict(weights)
    modnet.eval()

    # inference images

     

    for im_path in tqdm(glob.glob(os.path.join(image_dir,'*_detailed.png'))):
        # print('Processing:', im_path)

        matte_path = im_path.replace('_detailed.png', '_matting.png')
        
        im = Image.open(im_path)

        # unify image channels to 3
        im = np.asarray(im)
        if len(im.shape) == 2:
            im = im[:, :, None]
        if im.shape[2] == 1:
            im = np.repeat(im, 3, axis=2)
        elif im.shape[2] == 4:
            im = im[:, :, 0:3]

        # convert image to PyTorch tensor
        im = Image.fromarray(im)
        im = im_transform(im)

        # add mini-batch dim
        im = im[None, :, :, :]

        # resize image for input
        im_b, im_c, im_h, im_w = im.shape
        if max(im_h, im_w) < ref_size or min(im_h, im_w) > ref_size:
            if im_w >= im_h:
                im_rh = ref_size
                im_rw = int(im_w / im_h * ref_size)
            elif im_w < im_h:
                im_rw = ref_size
                im_rh = int(im_h / im_w * ref_size)
        else:
            im_rh = im_h
            im_rw = im_w
        
        im_rw = im_rw - im_rw % 32
        im_rh = im_rh - im_rh % 32
        im = F.interpolate(im, size=(im_rh, im_rw), mode='area')

        # inference
        _, _, matte = modnet(im.cuda() if torch.cuda.is_available() else im, True)

        # resize and save matte
        matte = F.interpolate(matte, size=(im_h, im_w), mode='area')
        matte = matte[0][0].data.cpu().numpy()

 
        # check if directory exists
        #print(os.path.dirname(matte_path))

        os.makedirs(os.path.dirname(matte_path), exist_ok=True)


        #print('save matte to {0}'.format(matte_path))

        Image.fromarray(((matte * 255).astype('uint8')), mode='L').save(os.path.join(matte_path))
        #exit()

        matting_array = np.array(Image.open(matte_path))
        rgb_array = np.array(Image.open(im_path))
        rgba = np.concatenate([rgb_array, matting_array[:, :, None]], axis=2)
         
        Image.fromarray(rgba).save(im_path.replace('_detailed.png', '_rgba.png'))
