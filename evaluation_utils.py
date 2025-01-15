"""
Evaluation metrics
Borrowed from HPLFlowNet
Date: May 2020

@inproceedings{HPLFlowNet,
  title={HPLFlowNet: Hierarchical Permutohedral Lattice FlowNet for
Scene Flow Estimation on Large-scale Point Clouds},
  author={Gu, Xiuye and Wang, Yijie and Wu, Chongruo and Lee, Yong Jae and Wang, Panqu},
  booktitle={Computer Vision and Pattern Recognition (CVPR), 2019 IEEE International Conference on},
  year={2019}
}
"""

import numpy as np


def evaluate_3d(sf_pred, sf_gt, scale=0.9105):

    # sf_gt_norm = np.linalg.norm(sf_gt, axis=-1) * scale
    # sf_pred_norm = np.linalg.norm(sf_pred, axis=-1) * scale
    #
    # mask = np.logical_or(sf_gt_norm > 1., sf_pred_norm > 1.)
    # sf_pred = sf_pred[mask]
    # sf_gt = sf_gt[mask]

    l2_norm = np.linalg.norm(sf_gt - sf_pred, axis=-1) * scale
    sf_gt_norm = np.linalg.norm(sf_gt, axis=-1) * scale

    EPE3D = l2_norm.mean()

    relative_err = l2_norm / (sf_gt_norm + 1e-10)

    acc3d_strict = (np.logical_or(l2_norm < 0.2, relative_err < 0.05)).astype(np.float).mean()
    acc3d_relax = (np.logical_or(l2_norm < 0.4, relative_err < 0.1)).astype(np.float).mean()
    outlier = (np.logical_or(l2_norm > 1.2, relative_err > 0.2)).astype(np.float).mean()

    return EPE3D, acc3d_strict, acc3d_relax, outlier


def evaluate_2d(flow_pred, flow_gt):
    """
    flow_pred: (N, 2)
    flow_gt: (N, 2)
    """

    epe2d = np.linalg.norm(flow_gt - flow_pred, axis=-1)
    epe2d_mean = epe2d.mean()

    flow_gt_norm = np.linalg.norm(flow_gt, axis=-1)
    relative_err = epe2d / (flow_gt_norm + 1e-5)

    acc2d = (np.logical_or(epe2d < 3., relative_err < 0.05)).astype(np.float).mean()

    return epe2d_mean, acc2d
