import argparse
import os
import pickle as pk

import cv2
import numpy as np
import torch
import torchvision.models as models
from easydict import EasyDict as edict
from hybrik.models import builder
from hybrik.utils.config import update_config
from hybrik.opt import cfg, logger, opt
from torch.utils.tensorboard import SummaryWriter
from hybrik.datasets import MixDataset, PW3D, HP3D, AGORAX
from hybrik.utils.transforms import flip, get_func_heatmap_to_coord
from torchvision import transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from tqdm import tqdm
from hybrik.utils.metrics import DataLogger, calc_coord_accuracy
from torchsummary import summary

# Configuration and pretrained model paths
cfg_file = './configs/smplx/test.yaml'
# CKPT = './pretrained_models/hybrikx_hrnet.pth'

parser = argparse.ArgumentParser(description='New training script')
# Load the configuration
cfg = update_config(cfg_file)
# opt = parser.parse_args()

device = "cpu"


# Load the pretrained detection model
# det_model = fasterrcnn_resnet50_fpn(pretrained=True)

def train(opt, train_loader, m, criterion, optimizer):
    loss_logger = DataLogger()
    acc_uvd_29_logger = DataLogger()
    acc_xyz_17_logger = DataLogger()
    m.train()
    hm_shape = cfg.MODEL.get('HEATMAP_SIZE')
    depth_dim = cfg.MODEL.EXTRA.get('DEPTH_DIM')
    hm_shape = (hm_shape[1], hm_shape[0], depth_dim)
    # root_idx_17 = train_loader.dataset.root_idx_17

    if opt.log:
        print('Training is true')
        train_loader = tqdm(train_loader, dynamic_ncols=True)

    for i, (inps, labels, _, bboxes) in enumerate(train_loader):
        if isinstance(inps, list):
            if device == "gpu":
                inps = [inp.cuda(opt.gpu).requires_grad_() for inp in inps]
            else:
                inps = [inp.cpu().requires_grad_() for inp in inps]
        else:
            if device == "gpu":
                inps = inps.cuda(opt.gpu).requires_grad_()
            else:
                inps = inps.cpu().requires_grad_()

        for k, _ in labels.items():
            if device == "gpu":
                labels[k] = labels[k].cuda(opt.gpu)
            else:
                labels[k] = labels[k].cpu()

        # print('labels: ', labels.keys())

        trans_inv = labels.pop('trans_inv')
        intrinsic_param = labels.pop('intrinsic_param')
        root = labels.pop('joint_root')
        depth_factor = labels.pop('depth_factor')

        # torch.autograd.set_detect_anomaly(True)
        # output = m(inps, trans_inv, intrinsic_param, root, depth_factor, None)
        output = m(inps, None)

        loss = criterion(output, labels)

        pred_uvd_jts = output.pred_uvd_jts
        pred_xyz_jts_17 = output.pred_xyz_jts_17
        label_masks_29 = labels['target_weight_29']
        label_masks_17 = labels['target_weight_17']
        if pred_uvd_jts.shape[1] != labels['target_uvd_29'].shape[1]:
            pred_uvd_jts = pred_uvd_jts.cpu().reshape(pred_uvd_jts.shape[0], 24, 3)
            gt_uvd_jts = labels['target_uvd_29'].cpu().reshape(pred_uvd_jts.shape[0], 29, 3)[:, :24, :]
            gt_uvd_mask = label_masks_29.cpu().reshape(pred_uvd_jts.shape[0], 29, 3)[:, :24, :]
            acc_uvd_29 = calc_coord_accuracy(pred_uvd_jts, gt_uvd_jts, gt_uvd_mask, hm_shape, num_joints=24)
        else:
            acc_uvd_29 = calc_coord_accuracy(pred_uvd_jts.cpu(), labels['target_uvd_29'].cpu(), label_masks_29.cpu(),
                                             hm_shape, num_joints=29)
        acc_xyz_17 = calc_coord_accuracy(pred_xyz_jts_17.cpu(), labels['target_xyz_17'].cpu(), label_masks_17.cpu(),
                                         hm_shape, num_joints=17, root_idx=root_idx_17)

        if isinstance(inps, list):
            batch_size = inps[0].size(0)
        else:
            batch_size = inps.size(0)

        loss_logger.update(loss.item(), batch_size)
        acc_uvd_29_logger.update(acc_uvd_29, batch_size)
        acc_xyz_17_logger.update(acc_xyz_17, batch_size)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        opt.trainIters += 1
        if opt.log:
            # TQDM
            train_loader.set_description(
                'loss: {loss:.8f} | accuvd29: {accuvd29:.4f} | acc17: {acc17:.4f}'.format(
                    loss=loss_logger.avg,
                    accuvd29=acc_uvd_29_logger.avg,
                    acc17=acc_xyz_17_logger.avg)
            )
    if opt.log:
        train_loader.close()

    return loss_logger.avg, acc_xyz_17_logger.avg


def validate_gt(m, opt, cfg, dataset, heatmap_to_coord, batch_size=32):
    gt_val_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=10, drop_last=False, pin_memory=True)
    kpt_pred = {}
    m.eval()
    # tot_err = 0
    hm_shape = cfg.MODEL.get('HEATMAP_SIZE')
    hm_shape = (hm_shape[1], hm_shape[0])

    for inps, labels, img_ids, bboxes in tqdm(gt_val_loader, dynamic_ncols=True):
        if device == "gpu":
            inps = inps.cuda(opt.gpu)
        else:
            inps = inps.cpu()

        trans_inv = labels.pop('trans_inv')
        intrinsic_param = labels.pop('intrinsic_param')

        root = labels.pop('joint_root')
        depth_factor = labels.pop('depth_factor')
        flip_output = labels.pop('is_flipped', None)

        output = m(inps, trans_inv, intrinsic_param, root, depth_factor, flip_output)

        pred_uvd_jts = output.pred_uvd_jts
        pred_xyz_jts_24 = output.pred_xyz_jts_24.reshape(inps.shape[0], -1, 3)[:, :24, :]
        pred_xyz_jts_24_struct = output.pred_xyz_jts_24_struct.reshape(inps.shape[0], 24, 3)
        pred_xyz_jts_17 = output.pred_xyz_jts_17.reshape(inps.shape[0], 17, 3)

        test_betas = output.pred_shape
        test_phi = output.pred_phi
        test_leaf = output.pred_leaf

        # Flip test
        if opt.flip_test:
            if isinstance(inps, list):
                inps_flip = [flip(inp) for inp in inps]
            else:
                inps_flip = flip(inps)

            output_flip = m(inps_flip, trans_inv, intrinsic_param,
                            root, depth_factor,
                            flip_item=(pred_uvd_jts, test_phi, test_leaf, test_betas),
                            flip_output=True)

            pred_uvd_jts_flip = output_flip.pred_uvd_jts

            pred_xyz_jts_24_flip = output_flip.pred_xyz_jts_24.reshape(
                inps.shape[0], -1, 3)[:, :24, :]
            pred_xyz_jts_24_struct_flip = output_flip.pred_xyz_jts_24_struct.reshape(
                inps.shape[0], 24, 3)
            pred_xyz_jts_17_flip = output_flip.pred_xyz_jts_17.reshape(
                inps.shape[0], 17, 3)

            pred_uvd_jts = pred_uvd_jts_flip

            pred_xyz_jts_24 = pred_xyz_jts_24_flip
            pred_xyz_jts_24_struct = pred_xyz_jts_24_struct_flip
            pred_xyz_jts_17 = pred_xyz_jts_17_flip

        pred_xyz_jts_24 = pred_xyz_jts_24.cpu().data.numpy()
        pred_xyz_jts_24_struct = pred_xyz_jts_24_struct.cpu().data.numpy()
        pred_xyz_jts_17 = pred_xyz_jts_17.cpu().data.numpy()
        pred_uvd_jts = pred_uvd_jts.cpu().data

        assert pred_xyz_jts_17.ndim in [2, 3]
        pred_xyz_jts_17 = pred_xyz_jts_17.reshape(
            pred_xyz_jts_17.shape[0], 17, 3)
        pred_uvd_jts = pred_uvd_jts.reshape(
            pred_uvd_jts.shape[0], -1, 3)
        pred_xyz_jts_24 = pred_xyz_jts_24.reshape(
            pred_xyz_jts_24.shape[0], 24, 3)
        pred_scores = output.maxvals.cpu().data[:, :29]

        for i in range(pred_xyz_jts_17.shape[0]):
            bbox = bboxes[i].tolist()
            pose_coords, pose_scores = heatmap_to_coord(
                pred_uvd_jts[i], pred_scores[i], hm_shape, bbox, mean_bbox_scale=None)
            kpt_pred[int(img_ids[i])] = {
                'xyz_17': pred_xyz_jts_17[i],
                'uvd_jts': pose_coords[0],
                'xyz_24': pred_xyz_jts_24[i]
            }

    with open(os.path.join(opt.work_dir, 'test_gt_kpt.pkl'), 'wb') as fid:
        pk.dump(kpt_pred, fid, pk.HIGHEST_PROTOCOL)

    tot_err_17 = dataset.evaluate_xyz_17(
        kpt_pred, os.path.join(opt.work_dir, 'test_3d_kpt.json'))
    _ = dataset.evaluate_uvd_24(
        kpt_pred, os.path.join(opt.work_dir, 'test_3d_kpt.json'))
    _ = dataset.evaluate_xyz_24(
        kpt_pred, os.path.join(opt.work_dir, 'test_3d_kpt.json'))
    # Additional evaluations for SMPL-X can be added here

    return tot_err_17


def main():
    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True)

    # print(opt)
    opt.gpu = 0
    logger.info('******************************')
    logger.info(opt)
    logger.info('******************************')
    logger.info(cfg)
    logger.info('******************************')
    # model initialization
    model = preset_model(cfg)

    if opt.params:
        print('Calculating model parameters and FLOPs')
        from thop import clever_format, profile
        if device == "gpu":
            input = torch.randn(1, 3, 256, 256).cuda(opt.gpu)
            flops, params = profile(model.cuda(opt.gpu), inputs=(input,))
        else:
            input = torch.randn(1, 3, 256, 256).cpu()
            flops, params = profile(model.cpu(), inputs=(input,))
        macs, params = clever_format([flops, params], "%.3f")
        logger.info(macs, params)

    if device == "gpu":
        model.cuda(opt.gpu)
    else:
        model.cpu()

    # Create the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.TRAIN.LR)
    # Create the loss function
    if device == "gpu":
        criterion = builder.build_loss(cfg.LOSS).cuda(opt.gpu)
    else:
        criterion = builder.build_loss(cfg.LOSS).cpu()
    # Create the learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=cfg.TRAIN.LR_STEP, gamma=cfg.TRAIN.LR_FACTOR)
    # Create the dataset
    train_dataset = None
    if cfg.DATASET.DATASET == 'mix_smplx_all':

        if not os.path.exists('train_dataset.pt'):
            print("Creating train_dataset.pt file")
            train_dataset = AGORAX(
                cfg=cfg,
                ann_file=cfg.DATASET.SET_LIST[0].TRAIN_SET,
                train=True,
                finetune=True,
                lazy_import=True
            )
            train_dataset = torch.save(train_dataset, 'train_dataset.pt')
        else:
            print("Loading train_dataset from train_dataset file")
            train_dataset = torch.load('train_dataset.pt')
    else:
        raise NotImplementedError
    # Create the tensorboard writer
    if opt.log:
        writer = SummaryWriter('.tensorboard/{}/{}-{}'.format(cfg.DATASET.DATASET, cfg.FILE_NAME, opt.exp_id))
    else:
        writer = None

    # Create the data loader
    # print('train_dataset: ', train_dataset)
    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True)
    # train_loader = torch.save(train_loader, 'train_loader.pt')
    train_loader = torch.load('train_loader.pt')
    # print('train_loader: ', list(train_loader)

    # gt val dataset
    if not os.path.exists('validation_dataset.pt'):
        print("Creating validation_dataset.pt file")
        gt_val_dataset_3dhp = AGORAX(
            cfg=cfg,
            ann_file=cfg.DATASET.SET_LIST[0].TEST_SET,
            train=False,
            finetune=True,
            lazy_import=True
        )
        gt_val_dataset_3dhp = torch.save(gt_val_dataset_3dhp, 'validation_dataset.pt')
    else:
        print("Loading train_dataset from train_dataset file")
        label = torch.load('validation_dataset.pt')

    heatmap_to_coord = get_func_heatmap_to_coord(cfg)

    opt.trainIters = 0
    best_err_h36m = 999
    # best_err_3dpw = 999

    for epoch in range(cfg.TRAIN.BEGIN_EPOCH, cfg.TRAIN.END_EPOCH):
        current_lr = optimizer.state_dict()['param_groups'][0]['lr']
        print(f'############# Starting Epoch {epoch} | LR: {current_lr} #############')

        # Training
        # Assuming `train` is a function that trains your model for one epoch and returns the loss
        loss = train(opt, train_loader, model, criterion, optimizer)
        print(f'Epoch: {epoch}, Loss: {loss}')

        # Step the LR scheduler
        lr_scheduler.step()

        # Save model checkpoint periodically
        if (epoch + 1) % cfg.SAVE_INTERVAL == 0:
            torch.save(model.state_dict(), f'./checkpoints/model_epoch_{epoch}.pth')

            with torch.no_grad():
                gt_tot_err_h36m = validate_gt(model, opt, cfg, gt_val_dataset_3dhp, heatmap_to_coord)
                if opt.log:
                    if gt_tot_err_h36m <= best_err_h36m:
                        best_err_h36m = gt_tot_err_h36m
                        torch.save(model.module.state_dict(),
                                   './data/{}/{}-{}/best_h36m_model.pth'.format(cfg.DATASET.DATASET, cfg.FILE_NAME,
                                                                                opt.exp_id))

                    logger.info(f'##### Epoch {opt.epoch} | h36m err: {gt_tot_err_h36m} / {best_err_h36m} #####')

        # Optional: Validation step here

    # Save the final model
    torch.save(model.state_dict(), './checkpoints/model_final.pth')
    print('Finished Training')


def preset_model(cfg):
    # model = torch.load(cfg.MODEL.PRETRAINED)
    # summary(model, (3, 256, 256))
    print(cfg.MODEL)
    model = builder.build_sppe(cfg.MODEL)
    # print("builder being called")
    # model.load_state_dict(torch.load(cfg.MODEL.PRETRAINED))
    if cfg.MODEL.PRETRAINED:
        print("Loading pretrained model")
        logger.info(f'Loading model from {cfg.MODEL.PRETRAINED}...')
        model.load_state_dict(torch.load(cfg.MODEL.PRETRAINED))
    # elif cfg.MODEL.TRY_LOAD:
    #     logger.info(f'Loading model from {cfg.MODEL.TRY_LOAD}...')
    #     pretrained_state = torch.load(cfg.MODEL.TRY_LOAD)
    #     model_state = model.state_dict()
    #     pretrained_state = {k: v for k, v in pretrained_state.items()
    #                         if k in model_state and v.size() == model_state[k].size()}
    #
    #     model_state.update(pretrained_state)
    #     model.load_state_dict(model_state)
    else:
        logger.info('Create new model')
        logger.info('=> init weights')
        model._initialize()

    return model


if __name__ == '__main__':
    main()
