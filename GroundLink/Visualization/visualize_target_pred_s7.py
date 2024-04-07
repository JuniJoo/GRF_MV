# Specify motion and force, display in aitviewer

import os

import numpy as np

from aitviewer.configuration import CONFIG as C
from aitviewer.renderables.point_clouds import PointClouds
from aitviewer.renderables.smpl import SMPLSequence
from aitviewer.viewer import Viewer
from aitviewer.renderables.spheres import Spheres


from aitviewer.renderables.lines import Lines
from aitviewer.renderables.arrows import Arrows
from aitviewer.renderables.lines import Lines
from aitviewer.renderables.arrows import Arrows
import matplotlib.pyplot as plt

import math as m
import torch

import util

# Load File Paths
Basepath = "../Data/moshpp/"
PredPath = "../GRF/ProcessedData/"
participant = 's007'
trial = 's007_20220705_tennisgroundstroke_1'
threshold = 0.1
fps = 250.0
ckp = 'pretrained_s7_noshape'
# End of Loading File Paths

# Basepath = "../Data/moshpp/"
# PredPath = "../GRF/SampleData/"
# participant = 's007'
# trial = 's007_20220705_hopping_1'
# threshold = 0.1
# fps = 250.0
# ckp = 'noshape_s7_3e6_73_3e-6'


Testing = True
sourcemotion = os.path.join(Basepath+participant, trial + '_stageii.npz')
print(sourcemotion)
# sourcemotion = os.path.join(Basepath + 'guy.npz')
if Testing:
    gt_file = os.path.join(PredPath+util.participants[participant]+'/test', trial+'.pth')
    print(gt_file)
    # gt_file = os.path.join(PredPath+util.participants[participant]+'/test', trial+'.pth')
else:
    gt_file = os.path.join(PredPath+util.participants[participant]+'/preprocessed', trial+'.pth')
    print(gt_file)
predicted = os.path.join(PredPath+util.participants[participant]+'/prediction/' + ckp, trial+'.pth')
print(predicted)


if __name__ == "__main__":
    # Load an AMASS sequence and make sure it's sampled at 60 fps. 
    # This loads the SMPL-X model.
    # We set transparency to 0.5 and render the joint coordinates systems.
    c = (149 / 255, 85 / 255, 149 / 255, 0.5)
    color_gt = (83 / 255, 189 / 255, 255 / 255, 1.0)
    color_pred = (255 / 255, 130 / 255, 53/255, 1.0)

    mesh = (102/255,102/255,102/255,0.5)
    fp_color = (127/255,127/255,128/255,1)
    # seq_amass = SMPLSequence.from_amass(
    #     npz_data_path=sourcemotion,
    #     fps_out=fps,
    #     color=mesh,
    #     name=trial,
    #     show_joint_angles=True,
    # )

    # ptc_amass = PointClouds(seq_amass.vertices, position=np.array([-1.0, 0.0, 0.0]), color=c, z_up=True)

    line_strip = util.get_fp()
    line_renderable = Lines(line_strip, color = fp_color, mode="lines")

    CoP, CoP_pred, GRF, GRF_pred = util.get_data_pred(gt_file, predicted, threshold)


    arrow_renderables = Arrows(
                CoP.numpy(),
                CoP.numpy()+GRF.numpy(),
                color= color_gt,
                is_selectable=True,
            )
    
    arrow_renderables_pred = Arrows(
                CoP_pred.numpy(),
                CoP_pred.numpy()+GRF_pred.numpy(),
                color= color_pred,
                is_selectable=True,
            )
    start_points = CoP.numpy()  # Fill in with your actual data
    end_points = CoP.numpy()+GRF.numpy()  # Fill in with your actual data

    # Calculate the displacement vectors (assuming end - start)
    displacement_vectors = end_points - start_points
    left_leg_data = displacement_vectors[:, 1, :]
    # print(left_leg_data)

    # Calculate the magnitudes of these vectors
    # This assumes the vectors represent force directly; if they represent displacement, additional info is needed
    magnitudes = np.linalg.norm(left_leg_data, axis=1)
    # If magnitudes are forces, they might already be in Newtons. If not, convert them:
    # For example, if magnitudes are in grams-force (gf), convert to Newtons (N) (1 gf ≈ 0.00980665 N)
    forces_in_newtons = magnitudes / 0.001
    forces_in_grams = forces_in_newtons / 4.44822

    from scipy.signal import savgol_filter

    # x = list(f.read().split(",\n"))
    # print(x)
    # x = [float(i) for i in x]
    # data = np.array(x)
    data_1 = [63.90001, 63.90001, 65.37722, 124.7542, 160.4593, 93.53595, 86.70101, 77.04022, 77.26635, 77.6586, 77.24857, 76.90285, 74.88904, 70.22436, 70.68954, 71.61938, 68.50549, 74.24326, 67.06897, 106.9514, 74.84806, 73.26868, 69.34991, 82.71538, 72.01599, 90.26415, 74.54601, 89.19671, 81.38461, 105.2954, 78.08477, 85.69209, 80.53862, 91.09465, 88.88157, 73.77534, 92.00191, 78.48871, 106.2897, 74.25578, 90.43433, 75.16794, 98.88834, 80.11853, 83.91777, 79.08041, 75.85466, 93.83932, 80.99273, 77.74304, 83.80462, 80.23857, 79.9077, 80.40704, 76.53968, 79.69623, 77.9474, 80.43078, 77.42458, 72.51418, 81.82337, 80.59399, 76.45032, 77.86549, 73.64225, 77.70918, 73.58325, 75.27, 74.84721, 74.24747, 75.32851, 72.20532, 73.32169, 72.92075, 75.82936, 71.79623, 75.53072, 71.9782, 75.74331, 73.52366, 72.20972, 74.58186, 76.00954, 72.60398, 75.7481, 69.42621, 77.92672, 71.77747, 86.64074, 75.59033, 75.78927, 75.84366, 75.67714, 76.45917, 77.9341, 76.51384, 78.75846, 77.18413, 76.81921, 80.22331, 78.80797, 76.71374, 78.56293, 78.94492, 79.92077, 79.30186, 80.06085, 80.71917, 80.83955, 83.12931, 80.32011, 81.22865, 86.45172, 79.25773, 82.98254, 75.76299, 97.34213, 88.84874, 81.26321, 85.80029, 86.74226, 74.8469, 73.40098, 74.79379, 75.64719, 75.83574, 79.22906, 74.56965, 73.62975, 86.92209, 75.58402, 80.31434, 79.17392, 77.92492, 78.78599, 76.01098, 79.4614, 77.28407, 80.9596, 76.44946, 78.77579, 77.29345, 79.8877, 74.96712, 78.41898, 78.10071, 77.76138, 81.57053, 77.27094, 73.86797, 88.90495, 101.596, 92.33276, 90.12785, 92.80228, 91.02515, 94.53485, 93.00729, 93.63256, 91.85243, 95.10468, 93.80038, 95.45055, 96.825, 95.10437, 92.13145, 86.86603, 115.9545, 105.1078, 96.41858, 103.5677, 87.00812, 117.3645, 99.28165, 114.449, 105.0423, 104.781, 102.0566, 110.4769, 110.228, 115.8925, 121.8673, 105.9374, 127.14, 107.6292, 129.106, 110.4745, 119.2436, 117.9527, 131.2887, 123.7441, 111.8768, 120.1677, 97.8702, 156.5629, 146.76, 120.5583, 114.0657, 149.7717, 128.089, 134.9404, 138.6816, 119.3022, 130.7353, 135.4041, 118.0771, 164.1588, 140.9535, 127.4352, 141.5981, 127.7329, 139.4755, 115.7413, 181.2654, 131.6492, 124.5995, 164.0477, 80.64438, 400.7914, 150.6249, 162.115, 139.2485, 154.0215, 145.2187, 138.0745, 142.8942, 156.7463, 167.1966, 143.4296, 109.0483, 219.0119, 134.0445, 144.5825, 163.7715, 134.1194, 148.3417, 135.0768, 155.8804, 128.7124, 142.2475, 138.8152, 0.0, 0.0, 0.0, 0.0, 0.0,  290.0867, 290.9481, 248.7685, 210.1484, 249.2943, 322.1109, 213.5777, 462.9674, 237.6219, 314.1939, 235.0032, 274.0852, 255.1113, 263.4041, 265.2102, 229.8289, 235.9674, 266.1481, 226.7815, 216.584, 228.8494, 217.9515, 210.8793, 168.174, 227.6014, 168.0552, 128.2458, 288.9757, 185.0123, 168.5258, 159.3155, 200.4305, 160.7718, 150.3278, 158.0811, 166.1051, 165.9452, 136.8611, 170.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,0.9758, 148.4104, 152.0374, 138.2664, 145.9147, 142.9459, 142.7333, 142.2357, 138.2876, 140.3671, 138.1223, 126.1366, 148.6563, 136.773, 126.019, 131.2408, 144.21, 123.5197, 126.5113, 124.9978, 142.871, 125.8588, 119.6543, 139.9007, 135.7707, 128.9422, 92.57687, 206.4863, 127.2813, 119.2342, 143.382, 126.8841, 139.7303, 131.1994, 123.1029, 146.9779, 125.5626, 122.9383, 148.2385, 125.5135, 139.502, 136.0429, 144.7229, 145.5811, 135.7453, 145.7632, 150.7273, 128.3927, 158.4961, 155.492, 154.7354, 134.272, 176.4565, 150.5507, 137.1192, 169.0163, 153.222, 180.1278, 162.9064, 151.5409, 170.9595, 174.6407, 168.6865, 106.1273, 345.0167, 151.2894, 181.8048, 181.916, 184.3015, 151.9166, 204.6472, 177.6407, 166.2941, 192.7715, 186.1848, 192.6223, 177.814, 190.7256, 203.3075, 188.1029, 177.1367, 186.4842, 219.2776, 198.7896, 181.3414, 226.3276, 171.8659, 223.5229, 217.8676, 187.5182, 220.1177, 186.7726, 215.7323, 254.3472, 171.2014, 245.755, 206.4427, 233.4814, 208.2121, 179.6589, 146.6821, 396.0265, 205.4226, 233.1962, 251.4648, 200.0593, 215.1983, 242.8615, 209.6326, 202.9277, 238.6026, 277.9584, 218.479, 188.4972, 260.0499, 222.8564, 227.3307, 236.3647, 223.0812, 249.3985, 226.3802, 207.3828, 240.1278, 218.6179, 231.8718, 239.6512, 234.869, 245.2271, 206.7774, 235.5656, 209.8626, 226.2049, 221.2195, 224.4662, 239.7189, 204.1394, 249.4595, 137.8899, 318.7697, 247.6738, 209.725, 227.0802, 172.0957, 261.474, 206.6263, 177.8797, 241.7041, 152.7993, 172.7368, 366.5152, 205.0648, 188.224, 217.872, 182.4462, 206.463, 198.6472, 193.1944, 198.5497, 175.8215, 188.4257, 184.315, 188.2067, 187.9007, 165.3174, 187.1389, 172.4413, 153.6874, 160.1436, 176.3349, 167.4016, 157.3454, 140.9158, 164.3165, 103.3227, 249.3915, 141.4694, 144.4552, 133.9665, 127.3512, 130.1164, 121.8176, 119.9138, 120.0164, 107.7292, 111.4538, 104.9269, 104.1772, 66.18524, 87.77724, 198.0376, 101.7358, 87.94476, 177.753, 133.5236, 137.451, 116.9427, 98.3253, 226.2751, 117.2724, 151.0447, 156.0819, 130.2986, 138.7955, 160.2463, 143.0787, 145.8956, 133.2442, 140.319, 141.8341, 125.8977, 150.0834, 130.3581, 128.1538, 155.2387, 125.948, 119.6641, 212.4111, 167.3303, 157.104, 131.2053, 160.3455, 132.2307, 161.8426, 133.9843, 156.0294, 140.6021, 139.9335, 170.2449, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 143.2759, 140.6226, 148.9368, 143.7042, 143.1979, 154.6478, 133.9469, 152.9875, 139.2514, 148.2659, 135.2885, 105.3173, 193.3394, 138.7935, 125.1714, 136.9918, 139.8717, 124.8592, 135.1043, 129.9558, 133.3934, 134.8878, 136.6275, 122.2666, 130.6792, 123.2556, 123.2779, 110.8292, 112.0811, 108.6147, 103.43, 108.8201, 100.1446, 99.02969, 99.55901, 101.8636, 87.70081, 131.282, 109.5113, 109.2376, 111.7336, 111.575, 108.5729, 103.2167, 99.6132, 138.755, 101.3186, 108.4564, 96.30521, 99.15716, 92.41939, 88.13443, 87.59026, 79.74176, 78.27759, 76.36302, 72.71306, 80.84494, 82.93558, 91.50937, 82.21947, 126.7272, 107.5546, 102.9983, 107.8384, 108.0267, 106.2816, 107.3138, 96.0282, 109.2332, 91.5546, 88.93719, 85.71146, 78.71104, 77.29084, 76.14458, 74.69948, 76.29629, 74.44199, 76.45486, 76.05573, 77.26381, 81.6601, 81.42292, 84.01399, 88.69222, 76.9108, 128.873, 133.1836, 86.46939, 68.24698, 112.8922, 73.3888, 81.02309, 83.83883, 85.5524, 103.2544, 146.1032, 152.5916, 177.9822, 173.1385, 167.9582, 148.6984, 111.0056, 103.474, 102.4893, 97.41029, 99.59303, 95.78514, 97.39383, 97.35822, 82.4703, 111.6718, 97.70728, 86.64957, 98.24255, 89.70344, 86.42162, 90.04769, 87.55388, 83.56839, 89.0812, 83.67582, 90.90005, 85.42962, 85.6881, 81.01189, 82.31835, 97.86024, 81.69335, 86.95707, 82.15837, 86.32037, 80.22184, 83.1431, 75.08366, 98.79309, 89.24133, 81.79057, 89.92517, 84.53399, 84.53764, 88.74113, 83.73449, 85.29799, 87.16787, 82.23168, 84.47522, 85.42609, 81.8493, 82.06789, 79.18706, 81.73339, 81.01165, 76.70094, 82.08068, 80.12447, 84.57778, 81.86017, 80.50595, 77.81855, 101.4312, 88.51517, 87.73733, 87.89722, 89.284, 91.02435, 90.3196, 91.92493, 94.46442, 93.969, 94.03835, 99.46505, 99.09503, 101.9504, 97.785, 109.2039, 104.1971, 115.4918, 112.8667, 125.1362, 111.4169, 123.9713, 135.0388, 126.6995, 106.7466, 213.2568, 155.33, 143.0653, 159.5111, 171.1602, 173.6381, 180.1954, 171.1759, 190.0936, 198.148, 206.348, 195.843, 196.929, 220.3882, 221.4689, 214.9828, 247.7607, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1519.83, 900.2168, 510.6193, 269.7682, 173.079, 300.9254, 305.557, 235.6969, 163.4488, 242.6493, 324.0037, 289.9978, 279.0435, 303.9168, 256.02, 232.7278, 280.571, 242.6021, 285.6453, 268.0098, 270.9538, 213.3746, 228.9567, 307.6253, 119.7452, 157.6667, 99.59586, 86.88142, 77.43978, 81.2601, 107.6937, 129.9234, 145.5204, 130.9158, 148.3411, 132.5131, 136.3093, 135.7593, 123.0403, 130.2066, 120.6543, 109.0108, 113.623, 112.4508, 114.273, 114.7377, 102.4059, 156.2634, 109.5792, 65.17312, 63.90001, 63.90001, 63.90001, 63.90001, 63.90001, 63.90001, 63.90001, 63.90001, 63.90001, 63.90001, 63.90001, 63.90001, 63.90001, 63.90001, 63.90001, 63.90001, 63.90001, 63.90001, 63.90001, 63.90001, 63.90001, 63.90001, 63.90001, 63.90001, 63.90001, 63.90001, 63.90001, 63.90001, 63.90001, 63.90001, 63.90001, 63.90001, 63.90001, 63.90001, 63.90001, 63.90001, 63.90001, 63.90001, 63.90001, 63.90001, 63.90001, 63.90001, 63.90001, 63.90001, 63.90001, 63.90001, 63.90001, 63.90001, 63.90001, 63.90001, 63.90001, 63.90001, 63.90001, 63.90001, 63.90001, 63.90001, 63.90001, 63.90001, 63.90001, 63.90001, 63.90001, 63.90001, 63.90001, 63.90001, 63.90001, 63.90001, 63.90001, 63.90001, 63.90001, 63.90001, 63.90001, 63.90001, 63.90001, 63.90001, 63.90001, 63.90001, 63.90001, 63.90001, 63.90001, 63.90001, 63.90001, 63.90001, 63.90001, 63.90001, 63.90001, 63.90001, 63.90001, 63.90001, 63.90001, 63.90001, 63.90001, 63.90001, 63.90001, 63.90001, 63.90001, 63.90001, 63.90001, 63.90001, 63.90001, 63.90001, 63.90001, 63.90001, 63.90001, 63.90001, 63.90001, 63.90001, 63.90001, 63.90001, 63.90001, 63.90001, 63.90001, 63.90001, 63.90001, 63.90001, 63.90001, 63.90001, 63.90001, 63.90001, 63.90001, 63.90001, 63.90001, 63.90001, 63.90001, 63.90001, 63.90001, 63.90001, 63.90001, 63.90001, 63.90001, 63.90001, 63.90001, 63.90001, 63.90001, 63.90001, 63.90001, 63.90001, 63.90001, 63.90001, 63.90001, 63.90001, 63.90001, 63.90001, 63.90001, 63.90001, 63.90001, 63.90001, 63.90001, 63.90001, 63.90001, 63.90001, 63.90001, 63.90001, 63.90001, 63.90001, 63.90001, 63.90001, 63.90001, 63.90001, 63.90001, 63.90001, 63.90001]
    yhat_1 = savgol_filter(data_1, 51, 3)  # window size 51, polynomial order 3
    #
    # mean_vals = data.mean(axis=0)
    # std_vals = data.std(axis=0)
    #
    # normalized_z_score = (x - mean_vals) / std_vals

    # plt.plot(data)
    # plt.plot(yhat_1, color='red')
    plt.ylabel('GRF(g)')
    plt.xlabel('Frame')
    plt.plot(forces_in_grams)
    plt.show()
    ppp


    # Display in the viewer.
    v = Viewer()
    v.run_animations = True
    v.scene.camera.position = np.array([10.0, 2.5, 0.0])
    v.scene.add(seq_amass)

    v.scene.add(line_renderable)
    v.scene.add(arrow_renderables)
    v.scene.add(arrow_renderables_pred)

    v.run()
