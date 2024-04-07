import sys
sys.path.append('../../UnderPressure')

import pickle
import models
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import glob
from torchsummary import summary

import time
from tqdm import tqdm

wild = {
    "S1" : False,
    "S2" : False,
    "S3" : False,
    "S4" : False,
    "S5" : False,
    "S6" : False,
    "S7" : False,
    "AMASS": True,
    "w074" : True,
	}

sub_mass = {
    "S1" : 69.81,
    "S2" : 66.68,
    "S3" : 53.07,
    "S4" : 71.67,
    "S5" : 90.7,
    "S6" : 48.99,
    "S7" : 63.96,
    "AMASS" : 80.0,
	}

#system = 'Windows'
system = 'Ubuntu'

save_img = False
save_high_res_img = True

ROOT = "../ProcessedData/"
subj = "S4"
mass = sub_mass[subj]
folder = "Male2MartialArtsKicks_c3d"
if wild[subj]:
    path = ROOT + subj + "/" + folder + "/preprocessed"
else:
    path = ROOT + subj + "/test"


filepath = os.path.join(path, "*.pth")
files = glob.glob(filepath)

k=20

checkpointname = 'pretrained_s4_noshape'
checkpointfile = '../checkpoint/' + checkpointname + '.tar'
pred_path = ROOT + subj + "/prediction/"
if not os.path.exists(pred_path):
    os.mkdir(pred_path)
if wild[subj]:
    pred_path_AMASS = ROOT + subj + "/" + folder + "/prediction/"
    if not os.path.exists(pred_path_AMASS):
        os.mkdir(pred_path_AMASS)

if system == 'Windows':
    bar = '\\'
else:
    bar = '/'

checkpoint = torch.load(checkpointfile)
model = models.DeepNetwork(state_dict=checkpoint["model"]).eval()
# model = model.summary()
print(model)
'''
Checkpoints are the saved model weights and biases. The model is loaded and the weights and biases are printed.
DeepNetwork(
    (0): Flatten(start_dim=2, end_dim=-1)
    (1): Transpose(-2, -1)
    (2): Dropout(p=0.0, inplace=False)
  (3): Conv1d(69, 128, kernel_size=(7,), stride=(1,), padding=(3,), padding_mode=replicate)
    (4): ELU(alpha=1.0)
    (5): Dropout(p=0.0, inplace=False)
  (6): Conv1d(128, 128, kernel_size=(7,), stride=(1,), padding=(3,), padding_mode=replicate)
    (7): ELU(alpha=1.0)
    (8): Dropout(p=0.0, inplace=False)
  (9): Conv1d(128, 256, kernel_size=(7,), stride=(1,), padding=(3,), padding_mode=replicate)
    (10): ELU(alpha=1.0)
    (11): Dropout(p=0.0, inplace=False)
  (12): Conv1d(256, 256, kernel_size=(7,), stride=(1,), padding=(3,), padding_mode=replicate)
    (13): ELU(alpha=1.0)
    (14): Transpose(-2, -1)
    (15): Dropout(p=0.2, inplace=False)
  (16): Linear(in_features=256, out_features=256, bias=True)
    (17): ELU(alpha=1.0)
    (18): Dropout(p=0.2, inplace=False)
  (19): Linear(in_features=256, out_features=256, bias=True)
    (20): ELU(alpha=1.0)
    (21): Dropout(p=0.2, inplace=False)
    (22): Linear(in_features=256, out_features=12, bias=False)
  (23): Unflatten(dim=-1, unflattened_size=(2, 6))
  
)
3.weight torch.Size([128, 69, 7])
3.bias torch.Size([128])
6.weight torch.Size([128, 128, 7])
6.bias torch.Size([128])
9.weight torch.Size([256, 128, 7])
9.bias torch.Size([256])
12.weight torch.Size([256, 256, 7])
12.bias torch.Size([256])
16.weight torch.Size([256, 256])
16.bias torch.Size([256])
19.weight torch.Size([256, 256])
19.bias torch.Size([256])
22.weight torch.Size([12, 256])

'''
# summary(model, (2, 69, 1))
#Given groups=1, weight of size [128, 69, 7], expected input[2, 165, 9] to have 69 channels, but got 165 channels instead

print("Sucessfully loaded model.")

pbar = tqdm(files)
pbar.set_description("Predicting: %s"%subj)
for file in pbar:
    trial = os.path.splitext(file)[0].split(bar)[-1]
    ref_data = torch.load(file)
    # for key in ref_data:
    #     #dict_keys(['gender', 'angles', 'trans', 'shape', 'framerate', 'to_global_rot', 'to_global', 'poses', 'CoP', 'GRF'])
    #     print(key, ref_data[key])

    # poses = ref_data["poses"]

    with open('please.pk', 'rb') as pickle_file:
        hybrik = pickle.load(pickle_file)
    poses_np = hybrik["pred_xyz"]
    print("poses_np", poses_np.shape)
    poses = torch.tensor(poses_np).float()

    print("size", poses.shape)
    trans = ref_data["trans"]

    with torch.no_grad():
        #TODO: so they do vGRF as well then try at somepoint
        #GRFs_pred = torch.Size([5836, 2, 6])
        GRFs_pred = model.vGRFs(poses.float().unsqueeze(0)).squeeze(0)
        # CoP_pred = model.contacts(poses.float().unsqueeze(0)).squeeze(0)
        # print("GRFs_pred", GRFs_pred.shape)
        # print("CoP_pred", CoP_pred.shape)
        # ppp
        # torch.Size([925, 2, 6])
        # print("GRFs_pred", GRFs_pred)
        file = '../../Visualization/hyrbikV.pk'
        with open(file, 'wb') as fid:
            pickle.dump(GRFs_pred, fid)
        # torch.save(GRFs_pred, file)
        # print(poses.float().unsqueeze(0))

    if not wild[subj]:
        post_process_path = pred_path + checkpointname
        if not os.path.exists(post_process_path):
            os.mkdir(post_process_path)
        # output_w_prediction = os.path.join(post_process_path, trial + ".pth")
        output_w_prediction = os.path.join(file + ".pth")


        weight = 9.81*mass

        output_pred = {}
        output_pred["GRF"] = ref_data["GRF"]
        output_pred["CoP"] = ref_data["CoP"]

        output_pred["prediction"] = GRFs_pred
        print("output_pred", output_pred.keys())
        ppp
        torch.save(output_pred, output_w_prediction)
    else:
        print("Going to else")
        ppp
        outputpath = pred_path_AMASS + checkpointname
        if not os.path.exists(outputpath):
            os.mkdir(outputpath)
        output = os.path.join(outputpath, trial + ".pth")
        torch.save(GRFs_pred, output)