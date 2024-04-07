# GRF_MV:Ground Reaction Force Estimation from Monocular Video

<a href="https://arxiv.org/abs/2304.05690">
    <img src='https://img.shields.io/badge/Paper-HybrIK--X-green?style=for-the-badge&logo=adobeacrobatreader&logoWidth=20&logoColor=white&labelColor=3CB371&color=40E0D0' alt='Paper PDF'>
</a>

## Installation instructions
Installing nessesary environment and dependencies from HybrIK-X.

``` bash
# 1. Create a conda virtual environment.
conda create -n GRF_MV python=3.8 -y
conda activate GRF_MV

# 2. Install PyTorch
conda install pytorch==1.9.1 torchvision==0.10.1 -c pytorch

# 4. Pull our code
git clone https://git.cs.bham.ac.uk/projects-2023-24/jxk010.git
cd jxk010

# 5. Install
pip install pycocotools
python setup.py develop  # or "pip install -e ."
```
GroundLink might require newer version of Python and Pytorch. Please refer to the [GroundLink](https://github.com/hanxingjian/GroundLink)


## Downloads
Directory under `$root/HybrIK` should look like this:

    ├── configs
    ├── examples
    ├── hybrik
    ├── hybrik.egg-info
    ├── model_files
    ├── pretrained_models
    ├── scripts
    └── setup.py

Within this, `pretrained_models` directory should contain the following files:

    └── hybrikx_hrnet.pth

Download necessary model files from [[Google Drive](https://drive.google.com/file/d/1un9yAGlGjDooPwlnwFpJrbGHRiLaBNzV/view?usp=sharing)] and un-zip them in the `${root/HybrIK}` directory.

HybrIK-X pretrained model is available for download from [[model]](https://drive.google.com/file/d/1bKIPD60z_Im4S3W2-rew6YtOtUGff6-v/view?usp=sharing) and its configuration [[cfg]](configs/smplx/256x192_hrnet_smplx_kid.yaml).

Directory under `$root/GroundLink` should look like this:

    ├── Visualization  
    │   ├── aitviewer
    │   └── models              
    ├── GRF                     
    │   ├── checkpoints         
    │   ├── Data               
    │       ├── fbx       
    │       ├── Force 
    │       ├── moshpp
    │       ├── AMASS
    │   ├── ProcessedData 
    │   └── scripts
    └── NN

Models can be downloaded from [SMPL-X](https://smpl-x.is.tue.mpg.de/) official website.

To install aitviewer (locally) with force plate coordinates setup:
```
cd (root)/GroundLink/Visualization
git clone git@github.com:eth-ait/aitviewer.git
mv forceplate.py aitviewer/
```

GroundLink Dataset can be downloaded from [GroundLink](https://csr.bu.edu/groundlink/). Also download [AMASS](https://amass.is.tue.mpg.de/) and place it under `Data/AMASS` directory.


    ├── HybrIK
    ├── GroundLink
    ├── README.md
    └── requirements.txt
GroundLink dataset needs to be downloaded from [GroundLink](https://github.com/hanxingjian/GroundLink) under the `$root/GroundLink` directory.

For Unity [Motion](https://bham-my.sharepoint.com/personal/jxk010_student_bham_ac_uk/Documents/Motions?csf=1&web=1&e=YFhQER)


## Running the demo
To produced 3D mesh recovered result from HybrIK-X
``` bash
cd HybrIK
python scripts/demo_video_x.py --video-name examples/dance.mp4 --out-dir res_dance --save-pk --save-img
```