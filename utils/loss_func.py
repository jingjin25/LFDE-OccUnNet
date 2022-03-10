
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import scipy as sp
import scipy.linalg as linalg
import math
from model.net_utils import warping, lf2sublfs
from utils.util import crop_boundary


def Weighted_Rec_Loss(lf, disp_coarse_sub, conf_coarse_sub, opt):

    # lf [N,a2,3,h,w]

    img_cv_gt = lf[:, opt.ind_ref].clone()  #[N,3,h,w]
    img_cv_gt = crop_boundary(img_cv_gt, opt.loss_crop)

    inds_all = torch.arange(opt.angular_num * opt.angular_num).view(opt.angular_num, opt.angular_num)
    inds_all = inds_all.view(1, -1, 1, 1, 1)
    inds_sub = lf2sublfs(inds_all).squeeze() #[1,4,an,1, 1,1]=>[4,an_sub]

    conf_coarse_sub = crop_boundary(conf_coarse_sub, opt.loss_crop)

    eps = 1e-6
    loss = 0.
    for i_sub in range(4):
        for k_s in inds_sub[i_sub]:
            img = warping(disp_coarse_sub[:, i_sub], k_s, opt.ind_ref, lf[:, k_s], opt.angular_num) #[N,h,w]
            img = crop_boundary(img, opt.loss_crop)
            diff = torch.add(img, -img_cv_gt)
            error = torch.sqrt(diff*diff + eps)
            loss += torch.sum(error * conf_coarse_sub[:, i_sub:i_sub+1].repeat(1,3,1,1)) / torch.numel(error)

    loss = loss / inds_sub.shape[1]

    return loss




def Conf_Prior_Loss(conf_coarse_sub, loss_crop):

    ###



def img_grads(I):
    I_dy = I[:, :, 1:, :] - I[:, :, :-1, :]
    I_dx = I[:, :, :, 1:] - I[:, :, :, :-1]
    return I_dx, I_dy

def Edge_Aware_Smoothness_Loss(D, I, edge_constant=150):

    img_gx_r, img_gy_r = img_grads(I[:, 0:1])
    img_gx_g, img_gy_g = img_grads(I[:, 1:2])
    img_gx_b, img_gy_b = img_grads(I[:, 2:3])

    weight_x = torch.exp(-edge_constant * (torch.abs(img_gx_r)+torch.abs(img_gx_g)+torch.abs(img_gx_b))/3.)
    weight_y = torch.exp(-edge_constant * (torch.abs(img_gy_r)+torch.abs(img_gy_g)+torch.abs(img_gy_b))/3.)

    disp_gx, disp_gy = img_grads(D)

    loss = (torch.mean(weight_x * torch.abs(disp_gx)) + torch.mean(weight_y * torch.abs(disp_gy)))/2.

    return loss
