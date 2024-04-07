import numpy as np
import os
import torch
import glob

import time
from tqdm import tqdm
from scipy.spatial.transform import Rotation

participants = {
    "s001": "S1",
    "s002": "S2",
    "s003": "S3",
    "s004": "S4",
    "s005": "S5",
    "s006": "S6",
    "s007": "S7",
}
#system = 'Windows'
system = 'Ubuntu'
motiontype = {
    'tree': 'yoga',
    'treearms': 'yoga',
    'chair': 'yoga',
    'squat': 'yoga',
    'worrier1': 'warrior',
    'worrier2': 'warrior',
    'sidestretch': 'side_stretch',
    'dog': 'hand',
    'jumpingjack': 'jump',
    'walk': 'walk',
    'walk_00': 'walk',
    'hopping': 'hopping',
    'ballethighleg': 'ballet_high',
    'balletsmalljump': 'ballet_jump',
    'whirl': 'dance',
    'lambadadance': 'yoga',
    'taichi': 'taichi',
    'step': 'stairs',
    'tennisserve': 'tennis',
    'tennisgroundstroke': 'tennis',
    'soccerkick': 'kicking',
    'idling': 'idling',
    'idling_00': 'idling',
    'static': 'static',
    'ballet_high_leg': 'ballet_high'
}

def parse_motion_force(sourcemotion, contactdata, outputfile):
    if os.path.exists(outputfile):
        print("File exists.. Skipping..")
        pass

    # load motion
    moshpp = np.load(sourcemotion, allow_pickle=True)
    # load force
    force_data = np.load(contactdata, allow_pickle=True)
    mocap = {}
    num_joints = 55
    num_body_joints = 22
    mocap["gender"] = moshpp["gender"]
    print("a")

    # load model file to remove pelvis offset from SMPL-X model
    modelpath = '../../Visualization/models/smplx/'
    # os.chdir(modelpath)
    # modelpath = '../../../Data/QTM_SOMA_MOSH/support_files/smplx/' + str(mocap["gender"])
    modelfile = os.path.join(modelpath, 'SMPLX_' + (str(mocap["gender"]).upper()) + '.npz')
    modeldata = np.load(modelfile, allow_pickle=True)
    print(list(modeldata.keys()))
    pelvis_offset = modeldata['f'][0]

    num_frames = min(len(moshpp['poses']), len(force_data.item()["CoP"]))
    mocap["angles"] = torch.reshape(torch.tensor(moshpp["poses"]), (len(moshpp['poses']), num_joints, 3))[:num_frames,
                      :num_body_joints, :]
    # mocap["angles"] = torch.index_select(mocap["angles"], 2, torch.LongTensor([0,2,1]))

    mocap["trans"] = torch.tensor(moshpp["trans"]).unsqueeze(1)[:num_frames] + pelvis_offset
    # mocap["trans"] = torch.index_select(mocap["trans"], 2, torch.LongTensor([0,2,1]))
    mocap["shape"] = torch.tensor(moshpp["betas"]).unsqueeze(1).repeat(num_frames, 1, 3)
    mocap["framerate"] = float(moshpp["mocap_framerate"])

    contact = {}

    COP = force_data.item()["CoP"][:num_frames]
    GRF = force_data.item()["GRF"][:num_frames]

    rotate_z = mocap["angles"][:, 0].clone()
    rotate = torch.zeros(num_frames, 3)
    rotate[:, 2] = rotate_z[:, 2]
    pelvis_rot = torch.tensor(Rotation.from_rotvec(rotate.numpy()).as_matrix())
    pelvis_t_project = mocap["trans"].clone()
    pelvis_t_project[:, :, 2] = 0.0

    transformation_mat = torch.eye(4).unsqueeze(0).repeat(num_frames, 1, 1)

    transformation_mat[:, :3, :3] = pelvis_rot
    mocap["to_global_rot"] = pelvis_rot
    rotation_mat_inv = torch.inverse(mocap["to_global_rot"])
    transformation_mat[:, :3, 3] = pelvis_t_project.squeeze(1)
    mocap["to_global"] = transformation_mat  # double tensor

    transformation_mat_inv = torch.inverse(transformation_mat)

    homo = torch.ones(num_frames, 2, 1)
    homo_COP = torch.cat((COP, homo), dim=-1)

    CoP_local = torch.matmul(transformation_mat_inv, homo_COP.transpose(-1, -2)).transpose(-1, -2)
    # GRF_local = torch.matmul(rotation_mat_inv, GRF.type('torch.DoubleTensor').transpose(-1, -2)).transpose(-1, -2)

    # shift CoP to projected pelvis
    contact["CoP"] = CoP_local[:, :, :-1].type('torch.FloatTensor')
    contact["GRF"] = GRF.type('torch.FloatTensor')

    homo_pelvis_one = torch.ones(num_frames, 1, 1)
    homo_pelvis = torch.cat((mocap["trans"], homo_pelvis_one), dim=-1).type('torch.FloatTensor')
    pelvis_local = torch.matmul(transformation_mat_inv, homo_pelvis.transpose(-1, -2)).transpose(-1, -2)

    mocap["poses"] = torch.cat((pelvis_local[:, :, :-1], mocap["angles"]), dim=1).type('torch.FloatTensor')

    torch.save(mocap | contact, outputfile)


for participant in participants:
    print("Participant ID: " + participant)
    cwd = os. getcwd()
    Datapath = "../../Data/"
    # mocap has npz format
    inputMocap = Datapath + 'moshpp/' + participant
    inputContact = Datapath + 'force/' + participant
    print(inputMocap)
    print(inputContact)

    datasetPath = '../ProcessedData/'
    outputPath = datasetPath + participants[participant] + '/preprocessed'
    print(outputPath)
    if not os.path.exists(outputPath):
        os.makedirs(outputPath)

    path = os.path.join(inputContact)
    os.chdir(path)
    forcefiles = glob.glob('**/*.npy', recursive=True)
    os.chdir(cwd)


    pbar = tqdm(forcefiles)
    pbar.set_description("Processing: %s" % participant)

    for forcefile in pbar:
        if system == 'Windows':
            bar = '\\'
        else:
            bar = '/'
        trial = os.path.splitext(forcefile)[0].split(bar)[-1]
        print(trial)
        motion = trial[14:-2]
        if motiontype[motion] == 'ballet_high':
            continue
        if participant == 's001' and motion == 'idling':
            continue
        outputfile = outputPath + '/' + trial + '.pth'
        print(outputfile)
        if os.path.exists(outputfile):
            print("Skipping: " + trial)
            continue
        else:
            sourcemotion = inputMocap + '/' + trial + "_stageii.npz"
            print(sourcemotion)
            sourceforce = inputContact + '/' + trial + '.npy'

            if not os.path.exists(sourcemotion):
                print(motion)
                print("motion file not exists.. Skipping...")
            else:
                parse_motion_force(sourcemotion, sourceforce, outputfile)

