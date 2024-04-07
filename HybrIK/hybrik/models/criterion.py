import math
import torch
import torch.nn as nn

from .builder import LOSS


amp = 1 / math.sqrt(2 * math.pi)


def weighted_l1_loss(input, target, weights, size_average):
    input = input * 64
    target = target * 64
    out = torch.abs(input - target)
    out = out * weights
    if size_average and weights.sum() > 0:
        return out.sum() / weights.sum()
    else:
        return out.sum()


def weighted_laplace_loss(input, sigma, target, weights, size_average):
    input = input
    target = target
    out = torch.log(sigma / amp) + torch.abs(input - target) / (math.sqrt(2) * sigma + 1e-5)
    out = out * weights
    if size_average and weights.sum() > 0:
        return out.sum() / weights.sum()
    else:
        return out.sum()


@LOSS.register_module
class L1LossDimSMPL(nn.Module):
    def __init__(self, ELEMENTS, size_average=True, reduce=True):
        super(L1LossDimSMPL, self).__init__()
        self.elements = ELEMENTS

        self.beta_weight = self.elements['BETA_WEIGHT']
        self.beta_reg_weight = self.elements['BETA_REG_WEIGHT']
        self.phi_reg_weight = self.elements['PHI_REG_WEIGHT']
        self.leaf_reg_weight = self.elements['LEAF_REG_WEIGHT']

        self.theta_weight = self.elements['THETA_WEIGHT']
        self.uvd24_weight = self.elements['UVD24_WEIGHT']
        self.xyz24_weight = self.elements['XYZ24_WEIGHT']
        self.xyz_smpl24_weight = self.elements['XYZ_SMPL24_WEIGHT']
        self.xyz_smpl17_weight = self.elements['XYZ_SMPL17_WEIGHT']
        self.vertice_weight = self.elements['VERTICE_WEIGHT']
        self.twist_weight = self.elements['TWIST_WEIGHT']

        self.criterion_smpl = nn.MSELoss()
        self.size_average = size_average
        self.reduce = reduce

    def phi_norm(self, pred_phis):
        assert pred_phis.dim() == 3
        norm = torch.norm(pred_phis, dim=2)
        _ones = torch.ones_like(norm)
        return self.criterion_smpl(norm, _ones)

    def leaf_norm(self, pred_leaf):
        assert pred_leaf.dim() == 3
        norm = pred_leaf.norm(p=2, dim=2)
        ones = torch.ones_like(norm)
        return self.criterion_smpl(norm, ones)

    def forward(self, output, labels):
        smpl_weight = labels['target_smpl_weight']

        # SMPL params
        loss_beta = self.criterion_smpl(output.pred_shape * smpl_weight, labels['target_beta'] * smpl_weight)
        loss_theta = self.criterion_smpl(output.pred_theta_mats * smpl_weight * labels['target_theta_weight'], labels['target_theta'] * smpl_weight * labels['target_theta_weight'])
        loss_twist = self.criterion_smpl(output.pred_phi * labels['target_twist_weight'], labels['target_twist'] * labels['target_twist_weight'])

        # Joints loss
        pred_uvd = output.pred_uvd_jts
        target_uvd = labels['target_uvd_29'][:, :pred_uvd.shape[1]]
        target_uvd_weight = labels['target_weight_29'][:, :pred_uvd.shape[1]]
        loss_uvd = weighted_l1_loss(output.pred_uvd_jts, target_uvd, target_uvd_weight, self.size_average)

        loss = loss_beta * self.beta_weight + loss_theta * self.theta_weight
        loss += loss_twist * self.twist_weight

        loss += loss_uvd * self.uvd24_weight

        return loss


@LOSS.register_module
class L1LossDimSMPLCam(nn.Module):
    def __init__(self, ELEMENTS, size_average=True, reduce=True):
        super(L1LossDimSMPLCam, self).__init__()
        self.elements = ELEMENTS

        self.beta_weight = self.elements['BETA_WEIGHT']
        self.beta_reg_weight = self.elements['BETA_REG_WEIGHT']
        self.phi_reg_weight = self.elements['PHI_REG_WEIGHT']
        self.leaf_reg_weight = self.elements['LEAF_REG_WEIGHT']

        self.theta_weight = self.elements['THETA_WEIGHT']
        self.uvd24_weight = self.elements['UVD24_WEIGHT']
        self.xyz24_weight = self.elements['XYZ24_WEIGHT']
        self.xyz_smpl24_weight = self.elements['XYZ_SMPL24_WEIGHT']
        self.xyz_smpl17_weight = self.elements['XYZ_SMPL17_WEIGHT']
        self.vertice_weight = self.elements['VERTICE_WEIGHT']
        self.twist_weight = self.elements['TWIST_WEIGHT']

        self.criterion_smpl = nn.MSELoss()
        self.size_average = size_average
        self.reduce = reduce

        self.pretrain_epoch = 40

    def phi_norm(self, pred_phis):
        assert pred_phis.dim() == 3
        norm = torch.norm(pred_phis, dim=2)
        _ones = torch.ones_like(norm)
        return self.criterion_smpl(norm, _ones)

    def leaf_norm(self, pred_leaf):
        assert pred_leaf.dim() == 3
        norm = pred_leaf.norm(p=2, dim=2)
        ones = torch.ones_like(norm)
        return self.criterion_smpl(norm, ones)

    def forward(self, output, labels, epoch_num=0):
        smpl_weight = labels['target_smpl_weight']

        # SMPL params
        loss_beta = self.criterion_smpl(output.pred_shape * smpl_weight, labels['target_beta'] * smpl_weight)
        loss_theta = self.criterion_smpl(output.pred_theta_mats * smpl_weight * labels['target_theta_weight'], labels['target_theta'] * smpl_weight * labels['target_theta_weight'])
        loss_twist = self.criterion_smpl(output.pred_phi * labels['target_twist_weight'], labels['target_twist'] * labels['target_twist_weight'])

        # Joints loss
        pred_xyz = (output.pred_xyz_jts_29)[:, :72]
        target_xyz = labels['target_xyz_24'][:, :pred_xyz.shape[1]]
        target_xyz_weight = labels['target_xyz_weight_24'][:, :pred_xyz.shape[1]]
        loss_xyz = weighted_l1_loss(pred_xyz, target_xyz, target_xyz_weight, self.size_average)

        batch_size = pred_xyz.shape[0]

        pred_uvd = output.pred_uvd_jts.reshape(batch_size, -1, 3)[:, :29]
        target_uvd = labels['target_uvd_29'][:, :29 * 3]
        target_uvd_weight = labels['target_weight_29'][:, :29 * 3]

        loss_uvd = weighted_l1_loss(
            pred_uvd.reshape(batch_size, -1),
            target_uvd.reshape(batch_size, -1),
            target_uvd_weight.reshape(batch_size, -1), self.size_average)

        loss = loss_beta * self.beta_weight + loss_theta * self.theta_weight
        loss += loss_twist * self.twist_weight

        loss += loss_uvd * self.uvd24_weight

        smpl_weight = (target_xyz_weight.sum(axis=1) > 3).float()
        smpl_weight = smpl_weight.unsqueeze(1)
        if 'cam_trans' in output.keys():
            pred_trans = output.cam_trans * smpl_weight
            target_trans = labels['camera_trans'] * smpl_weight
            trans_loss = self.criterion_smpl(pred_trans, target_trans)
            loss += (1 * trans_loss)

        pred_scale = output.cam_scale * smpl_weight
        target_scale = labels['camera_scale'] * smpl_weight
        scale_loss = self.criterion_smpl(pred_scale, target_scale)

        loss += (1 * scale_loss)

        return loss


@LOSS.register_module
class LaplaceLossDimSMPLCam(nn.Module):
    def __init__(self, ELEMENTS, size_average=True, reduce=True):
        super(LaplaceLossDimSMPLCam, self).__init__()
        self.elements = ELEMENTS

        self.beta_weight = self.elements['BETA_WEIGHT']
        self.beta_reg_weight = self.elements['BETA_REG_WEIGHT']
        self.phi_reg_weight = self.elements['PHI_REG_WEIGHT']
        self.leaf_reg_weight = self.elements['LEAF_REG_WEIGHT']

        self.theta_weight = self.elements['THETA_WEIGHT']
        self.uvd24_weight = self.elements['UVD24_WEIGHT']
        self.xyz24_weight = self.elements['XYZ24_WEIGHT']
        self.xyz_smpl24_weight = self.elements['XYZ_SMPL24_WEIGHT']
        self.xyz_smpl17_weight = self.elements['XYZ_SMPL17_WEIGHT']
        self.vertice_weight = self.elements['VERTICE_WEIGHT']
        self.twist_weight = self.elements['TWIST_WEIGHT']

        self.criterion_smpl = nn.MSELoss()
        self.size_average = size_average
        self.reduce = reduce

        self.pretrain_epoch = 40

    def phi_norm(self, pred_phis):
        assert pred_phis.dim() == 3
        norm = torch.norm(pred_phis, dim=2)
        _ones = torch.ones_like(norm)
        return self.criterion_smpl(norm, _ones)

    def leaf_norm(self, pred_leaf):
        assert pred_leaf.dim() == 3
        norm = pred_leaf.norm(p=2, dim=2)
        ones = torch.ones_like(norm)
        return self.criterion_smpl(norm, ones)

    def forward(self, output, labels, epoch_num=0):
        smpl_weight = labels['target_smpl_weight']

        # SMPL params
        loss_beta = self.criterion_smpl(output.pred_shape * smpl_weight, labels['target_beta'] * smpl_weight)
        loss_theta = self.criterion_smpl(output.pred_theta_mats * smpl_weight * labels['target_theta_weight'], labels['target_theta'] * smpl_weight * labels['target_theta_weight'])
        loss_twist = self.criterion_smpl(output.pred_phi * labels['target_twist_weight'], labels['target_twist'] * labels['target_twist_weight'])

        print("WOW YOU ARE HERE FINALLY IN LOSS TERM MAN WTF")
        '''
        output currently includes inps, trans_inv, intrinsic_param, root, depth_factor, flip_output
        (output needs to include feet_vertices)
        inps is inps.cuda(opt.gpu) -> I think it's just assinging gpu cuz the code was for multi-GPU
        trans_inv = labels.pop('trans_inv')
        intrinsic_param = labels.pop('intrinsic_param')
        root = labels.pop('joint_root')
        depth_factor = labels.pop('depth_factor')
        flip_output = labels.pop('is_flipped', None)

        feet_vertices(GT) = mean of right feet vertices from hp3d dataset
                which if joints_name_17 then 'L_Ankle' and 'R_Ankle'
                      if joints_name then 'left_foot' and 'right_foot'

        feet_vertices(pred) = output['feet_vertices'] #needs to be added to output i think

        feet_differences_list = [for i in feet_vertices(pred) - feet_vertices(GT)]
        for elem in feet_differences_list:
            if elem > 0:
                loss_feet = nn.MSELoss()
            else:
                loss_feet = nn.L1Loss() or nn.SmoothL1Loss() or 1000.0(float)

        loss += loss_feet * self.feet_weight (self.feet_weight ?????? what if they dont really have feet_weight)
        '''
        # loss_feet = self.criterion_smpl(output.pred_feet * labels['target_feet_weight'], labels['target_feet'] * labels['target_feet_weight'])

        # Joints loss
        pred_xyz = (output.pred_xyz_jts_29)[:, :72]
        # target_xyz = labels['target_xyz_24'][:, :pred_xyz.shape[1]]
        target_xyz_weight = labels['target_xyz_weight_24'][:, :pred_xyz.shape[1]]
        # loss_xyz = weighted_l1_loss(pred_xyz, target_xyz, target_xyz_weight, self.size_average)

        batch_size = pred_xyz.shape[0]

        pred_uvd = output.pred_uvd_jts.reshape(batch_size, -1, 3)[:, :29]
        pred_sigma = output.pred_sigma
        target_uvd = labels['target_uvd_29'][:, :29 * 3]
        target_uvd_weight = labels['target_weight_29'][:, :29 * 3]

        loss_uvd = weighted_laplace_loss(
            pred_uvd.reshape(batch_size, 29, -1),
            pred_sigma.reshape(batch_size, 29, -1),
            target_uvd.reshape(batch_size, 29, -1),
            target_uvd_weight.reshape(batch_size, 29, -1), self.size_average)

        loss = loss_beta * self.beta_weight + loss_theta * self.theta_weight
        loss += loss_twist * self.twist_weight

        loss += loss_uvd * self.uvd24_weight

        smpl_weight = (target_xyz_weight.sum(axis=1) > 3).float()
        smpl_weight = smpl_weight.unsqueeze(1)
        if 'cam_trans' in output.keys():
            pred_trans = output.cam_trans * smpl_weight
            target_trans = labels['camera_trans'] * smpl_weight
            trans_loss = self.criterion_smpl(pred_trans, target_trans)
            loss += (1 * trans_loss)

        pred_scale = output.cam_scale * smpl_weight
        target_scale = labels['camera_scale'] * smpl_weight
        scale_loss = self.criterion_smpl(pred_scale, target_scale)

        loss += (1 * scale_loss)

        return loss
