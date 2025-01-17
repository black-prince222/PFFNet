"""
Evaluation
Author: Wenxuan Wu
Date: May 2020
"""
import numpy as np
import torch.utils.data
import datetime
import logging

from tqdm import tqdm
from models import PointConvSceneFlowPWC8192selfglobalPointConv as PointConvSceneFlow
from models import multiScaleLoss
from pathlib import Path
from collections import defaultdict
import cmd_args

from datasets.facescape import FaceScapeDataSet
from main_utils import *
from utils import geometry
from evaluation_utils import evaluate_3d


def main():
    if 'NUMBA_DISABLE_JIT' in os.environ:
        del os.environ['NUMBA_DISABLE_JIT']

    global args
    args = cmd_args.parse_args_from_yaml(sys.argv[1])

    os.environ['CUDA_VISIBLE_DEVICES'] = '2,'

    '''CREATE DIR'''
    experiment_dir = Path('./Evaluate_experiment/')
    experiment_dir.mkdir(exist_ok=True)
    file_dir = Path(str(experiment_dir) + '/%sFlyingthings3d-' % args.model_name + str(
        datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')))
    file_dir.mkdir(exist_ok=True)
    checkpoints_dir = file_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = file_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)
    os.system('cp %s %s' % ('models.py', log_dir))
    os.system('cp %s %s' % ('pointconv_util.py', log_dir))
    os.system('cp %s %s' % ('evaluate_facescape.py', log_dir))
    os.system('cp %s %s' % ('config_evaluate_facescape.yaml', log_dir))

    '''LOG'''
    logger = logging.getLogger(args.model_name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(str(log_dir) + 'train_%s_sceneflow.txt' % args.model_name)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info('----------------------------------------TRAINING----------------------------------')
    logger.info('PARAMETER ...')
    logger.info(args)

    blue = lambda x: '\033[94m' + x + '\033[0m'
    model = PointConvSceneFlow()

    val_dataset = FaceScapeDataSet(
        train=False,
        npoints=args.num_points,
        root=args.data_root
    )
    logger.info('val_dataset: ' + str(val_dataset))
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True)

    # load pretrained model
    pretrain = args.pretrain
    model.load_state_dict(torch.load(pretrain))
    print('load model %s' % pretrain)
    logger.info('load model %s' % pretrain)

    model.cuda()

    epe3ds = AverageMeter()
    acc3d_stricts = AverageMeter()
    acc3d_relaxs = AverageMeter()
    outliers = AverageMeter()

    total_loss = 0
    total_seen = 0
    total_epe = 0
    metrics = defaultdict(lambda: list())
    for i, data in tqdm(enumerate(val_loader, 0), total=len(val_loader), smoothing=0.9):
        pos1, pos2, norm1, norm2, flow, _ = data

        # move to cuda
        pos1 = pos1.cuda()
        pos2 = pos2.cuda()
        norm1 = norm1.cuda()
        norm2 = norm2.cuda()
        flow = flow.cuda()

        model = model.eval()
        with torch.no_grad():
            pred_flows, fps_pc1_idxs, _, _, pred_flow_ori = model(pos1, pos2, norm1, norm2)
            loss = multiScaleLoss(pred_flows, flow[:, :4096, :], fps_pc1_idxs)

            full_flow = pred_flow_ori.permute(0, 2, 1)
            epe3d = torch.norm(full_flow - flow, dim=2).mean()

        total_loss += loss.cpu().data * args.batch_size
        total_epe += epe3d.cpu().data * args.batch_size
        total_seen += args.batch_size

        sf_np = flow.cpu().numpy()
        pred_sf = full_flow.cpu().numpy()

        EPE3D, acc3d_strict, acc3d_relax, outlier = evaluate_3d(pred_sf, sf_np)

        epe3ds.update(EPE3D)
        acc3d_stricts.update(acc3d_strict)
        acc3d_relaxs.update(acc3d_relax)
        outliers.update(outlier)

    mean_loss = total_loss / total_seen
    mean_epe = total_epe / total_seen
    str_out = '%s mean loss: %f mean epe: %f' % (blue('Evaluate'), mean_loss, mean_epe)
    print(str_out)
    logger.info(str_out)

    res_str = (' * EPE3D {epe3d_.avg:.4f}\t'
               'ACC3DS {acc3d_s.avg:.4f}\t'
               'ACC3DR {acc3d_r.avg:.4f}\t'
               'Outliers3D {outlier_.avg:.4f}\t'
        .format(
        epe3d_=epe3ds,
        acc3d_s=acc3d_stricts,
        acc3d_r=acc3d_relaxs,
        outlier_=outliers,
    ))

    print(res_str)
    logger.info(res_str)


if __name__ == '__main__':
    main()
