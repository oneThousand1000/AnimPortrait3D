# AnimPortrait3D 

This is the official code repository for our SIG'25 paper:  
> **Text-based Animatable 3D Avatars with Morphable Model Alignment**
>
> **ACM SIGGRAPH 2025 (Conference Track)**
>
> [Yiqian Wu](https://onethousandwu.com/), [Malte Prinzler](https://malteprinzler.github.io/), [Xiaogang Jin*](http://www.cad.zju.edu.cn/home/jin), [Siyu Tang](https://inf.ethz.ch/people/person-detail.MjYyNzgw.TGlzdC8zMDQsLTg3NDc3NjI0MQ==.html)

<div align="center">

[![Project](https://img.shields.io/badge/AnimPortrait3D-1?label=Project&color=8B93FF&logo=data:image/svg+xml;charset=utf-8;base64,PHN2ZyB0PSIxNzEyNDkwMTA3NzIxIiBjbGFzcz0iaWNvbiIgdmlld0JveD0iMCAwIDEwMjkgMTAyNCIgdmVyc2lvbj0iMS4xIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHAtaWQ9IjM4NzUiIHdpZHRoPSIyMDAiIGhlaWdodD0iMjAwIj48cGF0aCBkPSJNMTAwMS40MjMyMzggNDk0LjU5MnEyMS41MDQgMjAuNDggMjIuNTI4IDQ1LjA1NnQtMTYuMzg0IDQwLjk2cS0xOS40NTYgMTcuNDA4LTQ1LjA1NiAxNi4zODR0LTQwLjk2LTE0LjMzNnEtNS4xMi00LjA5Ni0zMS4yMzItMjguNjcydC02Mi40NjQtNTguODgtNzcuODI0LTczLjcyOC03OC4zMzYtNzQuMjQtNjMuNDg4LTYwLjQxNi0zMy43OTItMzEuNzQ0cS0zMi43NjgtMjkuNjk2LTY0LjUxMi0yOC42NzJ0LTYyLjQ2NCAyOC42NzJxLTEwLjI0IDkuMjE2LTM4LjQgMzUuMzI4dC02NS4wMjQgNjAuOTI4LTc3LjgyNCA3Mi43MDQtNzUuNzc2IDcwLjY1Ni01OS45MDQgNTUuODA4LTMwLjIwOCAyNy4xMzZxLTE1LjM2IDEyLjI4OC00MC45NiAxMy4zMTJ0LTQ0LjAzMi0xNS4zNnEtMjAuNDgtMTguNDMyLTE5LjQ1Ni00NC41NDR0MTcuNDA4LTQxLjQ3MnE2LjE0NC02LjE0NCAzNy44ODgtMzUuODR0NzUuNzc2LTcwLjY1NiA5NC43Mi04OC4wNjQgOTQuMjA4LTg4LjA2NCA3NC43NTItNzAuMTQ0IDM2LjM1Mi0zNC4zMDRxMzguOTEyLTM3Ljg4OCA4My45NjgtMzguNHQ3Ni44IDMwLjIwOHE2LjE0NCA1LjEyIDI1LjYgMjQuMDY0dDQ3LjYxNiA0Ni4wOCA2Mi45NzYgNjAuOTI4IDcwLjY1NiA2OC4wOTYgNzAuMTQ0IDY4LjA5NiA2Mi45NzYgNjAuOTI4IDQ4LjEyOCA0Ni41OTJ6TTQ0Ny40MzkyMzggMzQ2LjExMnEyNS42LTIzLjU1MiA2MS40NC0yNS4wODh0NjQuNTEyIDI1LjA4OHEzLjA3MiAzLjA3MiAxOC40MzIgMTcuNDA4bDM4LjkxMiAzNS44NHEyMi41MjggMjEuNTA0IDUwLjY4OCA0OC4xMjh0NTcuODU2IDUzLjI0OHE2OC42MDggNjMuNDg4IDE1My42IDE0Mi4zMzZsMCAxOTQuNTZxMCAyMi41MjgtMTYuODk2IDM5LjkzNnQtNDUuNTY4IDE4LjQzMmwtMTkzLjUzNiAwIDAtMTU4LjcycTAtMzMuNzkyLTMxLjc0NC0zMy43OTJsLTE5NS41ODQgMHEtMTcuNDA4IDAtMjQuMDY0IDEwLjI0dC02LjY1NiAyMy41NTJxMCA2LjE0NC0wLjUxMiAzMS4yMzJ0LTAuNTEyIDUzLjc2bDAgNzMuNzI4LTE4Ny4zOTIgMHEtMjkuNjk2IDAtNDcuMTA0LTEzLjMxMnQtMTcuNDA4LTM3Ljg4OGwwLTIwMy43NzZxODMuOTY4LTc2LjggMTUyLjU3Ni0xMzkuMjY0IDI4LjY3Mi0yNi42MjQgNTcuMzQ0LTUyLjczNnQ1Mi4yMjQtNDcuNjE2IDM5LjQyNC0zNi4zNTIgMTkuOTY4LTE4Ljk0NHoiIHAtaWQ9IjM4NzYiIGZpbGw9IiNmZmZmZmYiPjwvcGF0aD48L3N2Zz4=)](https://onethousandwu.com/animportrait3d.github.io/)
[![Paper](https://img.shields.io/badge/Paper%20(ACM%20Digital%20Library)-1?color=58A399&logo=data:image/svg+xml;charset=utf-8;base64,PHN2ZyB0PSIxNzEyNDkwMTQyMjM1IiBjbGFzcz0iaWNvbiIgdmlld0JveD0iMCAwIDEwMjQgMTAyNCIgdmVyc2lvbj0iMS4xIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHAtaWQ9IjU5MTQiIHdpZHRoPSIyMDAiIGhlaWdodD0iMjAwIj48cGF0aCBkPSJNODIzLjI5NiA2MC40MTZxNjUuNTM2IDAgOTkuMzI4IDM4LjR0MzMuNzkyIDkzLjY5NnY1NDMuNzQ0cTAgMjUuNi0yMS41MDQgNDYuMDhsLTE3MS4wMDggMTYzLjg0cS0xMy4zMTIgMTEuMjY0LTIyLjUyOCAxNC4zMzZ0LTIzLjU1MiAzLjA3MkgyNTguMDQ4cS0yMy41NTIgMC00Ny4xMDQtOS43Mjh0LTQxLjk4NC0yNy42NDgtMzAuMjA4LTQzLjAwOC0xMS43NzYtNTUuODA4di02MzQuODhxMC02MC40MTYgMzMuMjgtOTYuMjU2dDk0LjcyLTM1Ljg0aDU2OC4zMnogbS0yMTUuMDQgNjQyLjA0OHExMy4zMTIgMCAyMi41MjgtOS4yMTZUNjQwIDY3MC43MnEwLTE0LjMzNi05LjIxNi0yMy4wNHQtMjIuNTI4LTguNzA0SDI4Ny43NDRxLTEzLjMxMiAwLTIyLjUyOCA4LjcwNFQyNTYgNjcwLjcycTAgMTMuMzEyIDkuMjE2IDIyLjUyOHQyMi41MjggOS4yMTZoMzIwLjUxMnogbTEyOC0xOTIuNTEycTEzLjMxMiAwIDIyLjUyOC05LjIxNlQ3NjggNDc4LjIwOHQtOS4yMTYtMjIuNTI4LTIyLjUyOC05LjIxNkgyODcuNzQ0cS0xMy4zMTIgMC0yMi41MjggOS4yMTZUMjU2IDQ3OC4yMDh0OS4yMTYgMjIuNTI4IDIyLjUyOCA5LjIxNmg0NDguNTEyeiBtNjMuNDg4LTE5MS40ODhxMTMuMzEyIDAgMjIuNTI4LTkuMjE2dDkuMjE2LTIzLjU1MnEwLTEzLjMxMi05LjIxNi0yMi41Mjh0LTIyLjUyOC05LjIxNmgtNTEycS0xMy4zMTIgMC0yMi41MjggOS4yMTZUMjU2IDI4NS42OTZxMCAxNC4zMzYgOS4yMTYgMjMuNTUydDIyLjUyOCA5LjIxNmg1MTJ6IiBwLWlkPSI1OTE1IiBmaWxsPSIjZmZmZmZmIj48L3BhdGg+PC9zdmc+)]( )
[![Arxiv](https://img.shields.io/badge/Arxiv-1?color=A34343&logo=Arxiv)](https://arxiv.org/abs/2504.15835)
[![Supp](https://img.shields.io/badge/Supplementary-1?color=378CE7&logo=data:image/svg+xml;charset=utf-8;base64,PHN2ZyB0PSIxNzEyNDkwMTgyMzc1IiBjbGFzcz0iaWNvbiIgdmlld0JveD0iMCAwIDEwMjQgMTAyNCIgdmVyc2lvbj0iMS4xIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHAtaWQ9Ijg5MDIiIHdpZHRoPSIyMDAiIGhlaWdodD0iMjAwIj48cGF0aCBkPSJNNDg2LjQgNDA5LjZIMjgyLjE4NTE0M2E0MS4zOTg4NTcgNDEuMzk4ODU3IDAgMCAwLTQxLjc2NDU3MiA0MC45NmMwIDIyLjY3NDI4NiAxOC43MjQ1NzEgNDAuOTYgNDEuNzY0NTcyIDQwLjk2aDk2LjMyOTE0M2EzNjIuNzg4NTcxIDM2Mi43ODg1NzEgMCAwIDAtOTYuMzI5MTQzIDI0NS43NmMwIDk0LjU3MzcxNCAzNi42NDQ1NzEgMTgwLjUxNjU3MSA5Ni4zMjkxNDMgMjQ1Ljc2SDE1Ni43NDUxNDNBODIuNzk3NzE0IDgyLjc5NzcxNCAwIDAgMSA3My4xNDI4NTcgOTAxLjEyVjgxLjkyQzczLjE0Mjg1NyAzNi42NDQ1NzEgMTEwLjU5MiAwIDE1Ni43NDUxNDMgMGg2NjguNzQ1MTQzYzQ2LjA4IDAgODMuNjAyMjg2IDM2LjY0NDU3MSA4My42MDIyODUgODEuOTJ2MzgxLjE0NzQyOWEzODAuMTk2NTcxIDM4MC4xOTY1NzEgMCAwIDAtNDIyLjYxOTQyOC01My4zOTQyODZ6IG0yNTUuNDg4LTE2My44NGMwLTIyLjY3NDI4Ni0xOC43MjQ1NzEtNDAuOTYtNDEuODM3NzE0LTQwLjk2SDI4Mi4xMTJhNDEuMzk4ODU3IDQxLjM5ODg1NyAwIDAgMC00MS43NjQ1NzEgNDAuOTZjMCAyMi42NzQyODYgMTguNzI0NTcxIDQwLjk2IDQxLjc2NDU3MSA0MC45Nmg0MTcuOTM4Mjg2YzIzLjExMzE0MyAwIDQxLjgzNzcxNC0xOC4yODU3MTQgNDEuODM3NzE0LTQwLjk2ek02NTguMjg1NzE0IDQ1MC41NmMxNjEuNjQ1NzE0IDAgMjkyLjU3MTQyOSAxMjguMzY1NzE0IDI5Mi41NzE0MjkgMjg2LjcyUzgxOS45MzE0MjkgMTAyNCA2NTguMjg1NzE0IDEwMjRzLTI5Mi41NzE0MjktMTI4LjM2NTcxNC0yOTIuNTcxNDI4LTI4Ni43MiAxMzAuOTI1NzE0LTI4Ni43MiAyOTIuNTcxNDI4LTI4Ni43MnogbS0xMjUuMzY2ODU3IDMyNy42OGg4My42MDIyODZ2ODEuOTJjMCAyMi42NzQyODYgMTguNjUxNDI5IDQwLjk2IDQxLjc2NDU3MSA0MC45NiAyMy4xMTMxNDMgMCA0MS43NjQ1NzEtMTguMjg1NzE0IDQxLjc2NDU3Mi00MC45NnYtODEuOTJoODMuNjAyMjg1YzIzLjExMzE0MyAwIDQxLjgzNzcxNC0xOC4yODU3MTQgNDEuODM3NzE1LTQwLjk2IDAtMjIuNjc0Mjg2LTE4LjcyNDU3MS00MC45Ni00MS44Mzc3MTUtNDAuOTZINzAwLjA1MDI4NlY2MTQuNGMwLTIyLjY3NDI4Ni0xOC42NTE0MjktNDAuOTYtNDEuNzY0NTcyLTQwLjk2YTQxLjM5ODg1NyA0MS4zOTg4NTcgMCAwIDAtNDEuNzY0NTcxIDQwLjk2djgxLjkySDUzMi45MTg4NTdhNDEuMzk4ODU3IDQxLjM5ODg1NyAwIDAgMC00MS44Mzc3MTQgNDAuOTZjMCAyMi42NzQyODYgMTguNzI0NTcxIDQwLjk2IDQxLjgzNzcxNCA0MC45NnoiIGZpbGw9IiNmZmZmZmYiIHAtaWQ9Ijg5MDMiPjwvcGF0aD48L3N2Zz4=)](https://drive.google.com/file/d/1bt67uAtJyfh8ZAUw7fl7QvA0B5oyWZWe)


</div>


<div align="center">

[![Video](https://img.shields.io/badge/Video-1?color=E178C5&logo=data:image/svg+xml;charset=utf-8;base64,PHN2ZyB0PSIxNzEyNDkwMjU5OTgxIiBjbGFzcz0iaWNvbiIgdmlld0JveD0iMCAwIDEwMjQgMTAyNCIgdmVyc2lvbj0iMS4xIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHAtaWQ9IjEyOTE2IiBpZD0ibXhfbl8xNzEyNDkwMjU5OTgyIiB3aWR0aD0iMjAwIiBoZWlnaHQ9IjIwMCI+PHBhdGggZD0iTTYyMy4zMzAyOTE0ODQ0NDQ0IDIwMy44OTk4OTE0ODQ0NDQ0Mkg5Ny43NDc3NTI5NjAwMDAwMXMtODguMDI4NjAyNTk1NTU1NTUgMC04OC4wMjg2MDI1OTU1NTU1NSA4NS40Mzk1MjU1NDY2NjY2NnY0MzIuMzc1NzgyOTY4ODg4ODdjMCA4NS40Mzk1MjU1NDY2NjY2NiA4OC4wMjg2MDI1OTU1NTU1NSA4NS40Mzk1MjU1NDY2NjY2NiA4OC4wMjg2MDI1OTU1NTU1NSA4NS40Mzk1MjU1NDY2NjY2Nmg1MjUuNTgyNTM4NTI0NDQ0NHM4OC4wMjg2MDI1OTU1NTU1NSAwIDg4LjAyODYwMjU5NTU1NTU1LTg1LjQzOTUyNTU0NjY2NjY2VjI5MS45Mjg0OTQwODAwMDAwNmMwLTg4LjAyODYwMjU5NTU1NTU1LTg4LjAyODYwMjU5NTU1NTU1LTg4LjAyODYwMjU5NTU1NTU1LTg4LjAyODYwMjU5NTU1NTU1LTg4LjAyODYwMjU5NTU1NTU1ek05ODMuMjExOTMwNzM3Nzc3OCAyNDcuOTE0MTkyMjEzMzMzM2MtNy43NjcyMzAwMDg4ODg4ODgtMi41ODkwNzcwNDg4ODg4ODg2LTE1LjUzNDQ1ODg3OTk5OTk5OS0yLjU4OTA3NzA0ODg4ODg4ODYtMjAuNzEyNjExODQgMi41ODkwNzcwNDg4ODg4ODg2bC0xNzMuNDY4MTI5MjggMTM0LjYzMTk4MDM3MzMzMzMzYy01LjE3ODE1Mjk2IDUuMTc4MTUyOTYtNy43NjcyMzAwMDg4ODg4ODggMTAuMzU2MzA1OTItNy43NjcyMjg4NzExMTExMTE1IDE1LjUzNDQ1ODg3OTk5OTk5OXYyMTQuODkzMzUyOTYwMDAwMDJjMCA1LjE3ODE1Mjk2IDIuNTg5MDc3MDQ4ODg4ODg4NiAxMi45NDUzODI5Njg4ODg4ODggNy43NjcyMjg4NzExMTExMTE1IDE1LjUzNDQ2MDAxNzc3Nzc3N2wxNzMuNDY4MTI5MjggMTM0LjYzMTk3OTIzNTU1NTU4YzIuNTg5MDc3MDQ4ODg4ODg4NiAyLjU4OTA3NzA0ODg4ODg4ODYgNy43NjcyMzAwMDg4ODg4ODggNS4xNzgxNTI5NiAxMi45NDUzODE4MzExMTExMTIgNS4xNzgxNTQwOTc3Nzc3NzcgMi41ODkwNzcwNDg4ODg4ODg2IDAgNS4xNzgxNTI5NiAwIDEwLjM1NjMwNzA1Nzc3Nzc3OC0yLjU4OTA3NzA0ODg4ODg4ODYgNy43NjcyMzAwMDg4ODg4ODgtMi41ODkwNzcwNDg4ODg4ODg2IDEwLjM1NjMwNTkyLTEwLjM1NjMwNTkyIDEwLjM1NjMwNTkyLTE4LjEyMzUzNTkyODg4ODg4OFYyNjYuMDM3NzI4MTQyMjIyMjVjMC03Ljc2NzIzMDAwODg4ODg4OC01LjE3ODE1Mjk2LTE1LjUzNDQ1ODg3OTk5OTk5OS0xMi45NDUzODI5Njg4ODg4ODgtMTguMTIzNTM1OTI4ODg4ODg4eiIgZmlsbD0iI2ZmZmZmZiIgcC1pZD0iMTI5MTciPjwvcGF0aD48L3N2Zz4=)](https://youtu.be/UgNcuUKAc7A)
[![Github](https://img.shields.io/github/stars/oneThousand1000/AnimPortrait3D)](https://github.com/oneThousand1000/AnimPortrait3D)
[![dataset-on-hf](https://huggingface.co/datasets/huggingface/badges/raw/main/dataset-on-hf-sm.svg)](https://huggingface.co/datasets/onethousand/AnimPortrait3D_gallery)

</div>

![Representative_Image](./assets/teaser.jpg) 


##  Requirements

1. Python 3.10
2. CUDA>=12.1
3. At least 24 GB of memory 
4. Linux 
5. Tested on NVIDIA TITAN RTX GPU (3.5h per portrait)

## Installation

**Clone AnimPortrait3D to `/path/to/AnimPortrait3D`**

```
git clone git@github.com:oneThousand1000/AnimPortrait3D.git
```

**Create environment**
```
cd AnimPortrait3D
conda env create -f environment.yaml 
source activate AnimPortrait3D
```

**Run Install Script**
```
bash ./install.sh
```

### Prepare Data 

> Run `fetch_data.sh` to download all required files. Before proceeding, ensure you have registered accounts for [SMPLX](https://smpl-x.is.tue.mpg.de/), [SMPL](https://smpl.is.tue.mpg.de/), and [SMPLify](https://smplify.is.tue.mpg.de/), [FLAME](https://flame.is.tue.mpg.de/) to access the necessary models.

> **Note:** The script downloads pretrained diffusion models, ControlNet, and the Sapiens model to `/path/to/pretrained_model`. Please ensure that the specified path has sufficient storage space (> 50 GB).

```
bash ./fetch_data.sh \
    /path/to/pretrained_model \
    /path/to/AnimPortrait3D

# example:

bash ./fetch_data.sh \
    /path/to/AnimPortrait3D/pretrained_model \
    /path/to/AnimPortrait3D

``` 

> **AnimPortrait3D** builds upon the outputs of **Portrait3D** as its starting point. To generate your own starting avatar, please refer to [Portrait3D](https://github.com/oneThousand1000/Portrait3D). Alternatively, you can download pre-generated results from our [Hugging Face gallery](https://huggingface.co/datasets/onethousand/Portrait3D_gallery). For optimal performance, we recommend using a Portrait3D avatar with a neutral expression and without long hair.

> Our data preparation script (`fetch_data.sh`) also downloads an exemplar **Portrait3D** result to `outputs/Portrait3D_results`. Ensure that your **Portrait3D** results are organized in the following structure:

```
/path/to/Portrait3D_output
│
└─── {face_id}.pth [the pyramid trigrid file generated by Portrait3D]  
│
└─── {face_id}.txt [the prompt]
│
└─── {face_id}.png [preview image (optional)]
│   
└─── ...
```

##  3D Avatar Initialization (Sec 3.1)

> If you encounter any issues while running this project, please first check our [Q&A](https://github.com/oneThousand1000/AnimPortrait3D/blob/main/docs/Q&A.md) for possible solutions.

### Mesh Optimization

> The following script trains a high-quality 3D avatar mesh from the **Portrait3D** avatar.  
>  
> Please ensure you use absolute paths.  
>  
> The `face_id` corresponds to the filename in `/path/to/Portrait3D_output`. Use `000` to input the exemplar data.
```
bash 0_mesh_optimize.sh \
    face_id \
    /path/to/Portrait3D_output \
    /path/to/pretrained_model \
    /path/to/AnimPortrait3D_output

# example:

bash 0_mesh_optimize.sh \
    000 \
    /path/to/AnimPortrait3D/Portrait3D_output \
    /path/to/AnimPortrait3D/pretrained_model \
    /path/to/AnimPortrait3D/AnimPortrait3D_output

```

### Asset Mesh Generation

> The following script generates meshes for hair and clothing and fits an SMPL-X model to the **Portrait3D** avatar.

**face segmentation**

> We use **Sapiens** for face segmentation. Due to the complexity of its installation, we do not include it directly in our project. Instead, please follow the instructions in [Sapiens](https://github.com/facebookresearch/sapiens) to set up a separate environment.  
>  
> Next, place [`face_seg.sh`](https://github.com/oneThousand1000/AnimPortrait3D/blob/main/fetch_data.sh) in `./seg/scripts/demo/local/face_seg.sh` under the **Sapiens** project root. Then, run `face_seg.sh` **within the Sapiens environment** as follows:
```
face_seg.sh \
    face_id \
    /path/to/AnimPortrait3D_output \
    /path/to/pretrained_sapiens

# example

face_seg.sh \
    000 \
    /path/to/AnimPortrait3D/AnimPortrait3D_output \
    /path/to/AnimPortrait3D/pretrained_model/sapiens

```
> Note we have already downloaded sapiens models in `fetch_data.sh`, please specify /path/to/pretrained_sapiens as `/path/to/pretrained_model`/sapiens

**Asset mesh segment**

> Please ensure you use absolute paths. 
```
bash 1_asset_mesh_gen.sh \
    face_id \
    /path/to/Portrait3D_output \
    /path/to/pretrained_model \
    /path/to/AnimPortrait3D_output

# example

bash 1_asset_mesh_gen.sh \
    000 \
    /path/to/AnimPortrait3D/Portrait3D_output \
    /path/to/AnimPortrait3D/pretrained_model \
    /path/to/AnimPortrait3D/AnimPortrait3D_output

```

### Avatar Geometry and Appearance Initialization
> The following script trains a **3DGS** avatar with carefully initialized geometry and appearance.
> 
> Please ensure you use absolute paths. 
```
bash 2_avatar_initialization.sh \
    face_id \
    /path/to/Portrait3D_output \
    /path/to/pretrained_model \
    /path/to/AnimPortrait3D_output

# example

bash 2_avatar_initialization.sh \
    000 \
    /path/to/AnimPortrait3D/Portrait3D_output \
    /path/to/AnimPortrait3D/pretrained_model \
    /path/to/AnimPortrait3D/AnimPortrait3D_output


```

##  Dynamic Avatar Optimization (Sec 3.2)

> The following script trains a **3DGS** avatar with dynamic expressions and poses. The training process consists of two stages: first, we pre-train the eye and mouth regions separately, and then we train the full avatar.  
>  
> Please ensure you use absolute paths.   
>  
> Here, the **"abstract_prompt"** refers to a simple description that broadly categorizes the avatar, such as *"a boy"*, *"an old man"*, or *"a woman"*.

```
bash 3_dynamic_avatar_optimization.sh \
    face_id \
    /path/to/Portrait3D_output \
    /path/to/pretrained_model \
    /path/to/AnimPortrait3D_output \
    "abstract_prompt"

# example

bash 3_dynamic_avatar_optimization.sh \
    000 \
    /path/to/AnimPortrait3D/Portrait3D_output \
    /path/to/AnimPortrait3D/pretrained_model \
    /path/to/AnimPortrait3D/AnimPortrait3D_output \
    "a boy"

```
> This script also outputs the path of the generated final avatar and the fitted parameters, which are used for rendering and visualization.

## Inference and Visualization

We provide generated results at [HuggingFace](https://huggingface.co/onethousand/AnimPortrait3D_gallery), download and enjoy!

### Rendering Animated results

> Please specify the path to the final **PLY** file using `--points_cloud` and the path to the fitted parameters using `--fitted_parameters`.
>  
> For input expressions and poses, we provide two exemplars at `./GSAvatar/test_motion`. You can also use [VHAP](https://github.com/ShenhanQian/VHAP) to reconstruct motion from in-the-wild videos, or extract them from the [NerSemble](https://tobias-kirschstein.github.io/nersemble/) dataset. 
> 
> We provide some motion sequences reconstructed from [VFHQ](https://liangbinxie.github.io/projects/vfhq/) dataset using 
[VHAP](https://github.com/ShenhanQian/VHAP), please refer to [google drive](https://drive.google.com/file/d/1ZVQNUn8Kprlu9JviONSGMyrno7N2Ucdn/view?usp=drive_link) for download.
```
cd ./GSAvatar
python render_animation.py \
        --points_cloud=/path/to/points_cloud  \
        --fitted_parameters=/path/to/fitted_parameters  \
        --video_out_path=/path/to/output_video  \
        --exp_path=/path/to/expression_params \
        --pose_path=/path/to/poses
cd ..
```

### Interactive Rendering

```
cd ./GSAvatar
python local_viewer.py \
    --points_cloud=/path/to/points_cloud  \
    --fitted_parameters=/path/to/fitted_parameters
cd ..
```


https://github.com/user-attachments/assets/fe1417cb-6769-47d3-a286-74d41c1829d9



## Contact

[onethousand1250@gmail.com](mailto:onethousand1250@gmail.com)



## Citation

If you find this project helpful to your research, please consider citing:

```
@article{AnimPortrait3D_sig25,
      author = {Wu, Yiqian and Prinzler, Malte and Jin, Xiaogang and Tang, Siyu},
      title = {Text-based Animatable 3D Avatars with Morphable Model Alignment},
      year = {2025}, 
      isbn = {9798400715402}, 
      publisher = {Association for Computing Machinery},
      address = {New York, NY, USA},
      url = {https://doi.org/10.1145/3721238.3730680},
      doi = {10.1145/3721238.3730680},
      articleno = {},
      numpages = {11},
      location = {Vancouver, BC, Canada},
      series = {SIGGRAPH '25}
}
```



## Acknowledgements
We want to express our thanks to those in the open-source community for their valuable contributions.


 
## Usage Limitations Reminder

This project incorporates modified versions of the following projects:

- [gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting)
- [GaussianAvatars](https://github.com/ShenhanQian/GaussianAvatars)
- [pytorch-hair-segmentation](https://github.com/YBIGTA/pytorch-hair-segmentation)
- [Unique3D](https://github.com/AiuniAI/Unique3D)
- [LucidDreamer](https://github.com/EnVision-Research/LucidDreamer)

Please note the following limitations:

1. The project is provided strictly for non-commercial research and evaluation purposes.
2. All original copyright, patent, trademark, and attribution notices must remain intact.

By using or distributing this project, you agree to comply with the licensing conditions specified by each of the above projects.

 