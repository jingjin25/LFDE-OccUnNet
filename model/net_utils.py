import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import math


def lf2sublfs(lf):
    # print('lf', lf.shape)
    N, an2, c, h, w = lf.shape
    an = int(math.sqrt(an2))

    ind_cv = (an - 1) // 2
    # print('ind cv', ind_cv)

    lf_lu = lf.view(N, an, an, c, h, w)
    lf_ru = torch.flip(lf_lu, [2, 5])
    lf_ld = torch.flip(lf_lu, [1, 4])
    lf_rd = torch.flip(lf_lu, [1, 2, 4, 5])

    sub_lf_lu = lf_lu[:, :ind_cv+1, :ind_cv+1].contiguous().view(N, -1, c, h, w)
    sub_lf_ru = lf_ru[:, :ind_cv+1, :ind_cv+1].contiguous().view(N, -1, c, h, w)
    sub_lf_ld = lf_ld[:, :ind_cv+1, :ind_cv+1].contiguous().view(N, -1, c, h, w)
    sub_lf_rd = lf_rd[:, :ind_cv+1, :ind_cv+1].contiguous().view(N, -1, c, h, w)

    sub_lfs = torch.stack([sub_lf_lu, sub_lf_ru, sub_lf_ld, sub_lf_rd], 1)  # [N,4,an,3,h,w]

    return sub_lfs


def sub_spatial_flip(sub_imgs):
    # [N, 4, h, w]
    sub_imgs = torch.stack([sub_imgs[:, 0],
                            torch.flip(sub_imgs[:, 1], [2]),
                            torch.flip(sub_imgs[:, 2], [1]),
                            torch.flip(sub_imgs[:, 3], [1, 2]),
                            ], 1)
    return sub_imgs

def sub_spatial_flip_5D(sub_imgs):
    # [N, 4, c, h, w]
    sub_imgs = torch.stack([sub_imgs[:, 0],
                            torch.flip(sub_imgs[:, 1], [3]),
                            torch.flip(sub_imgs[:, 2], [2]),
                            torch.flip(sub_imgs[:, 3], [2, 3]),
                            ], 1)
    return sub_imgs


def warping(disp, ind_source, ind_target, img_source, an):
    '''warping one source image/map to the target'''
    # an angular number
    # disparity: int or [N,h,w]
    # ind_souce
    # ind_target
    # img_source [N,c,h,w]

    # ==> out [N,c,h,w]
    # print('img source ', img_source.shape)

    N, c, h, w = img_source.shape
    disp = disp.type_as(img_source)
    # ind_source = ind_source.type_as(disp)
    # ind_target = ind_target.type_as(disp)
    # print(img_source.shape)
    # coordinate for source and target
    # ind_souce = torch.tensor([0,an-1,an2-an,an2-1])[ind_source]
    ind_h_source = math.floor(ind_source / an)
    ind_w_source = ind_source % an

    ind_h_target = math.floor(ind_target / an)
    ind_w_target = ind_target % an

    # generate grid
    XX = torch.arange(0, w).view(1, 1, w).expand(N, h, w).type_as(img_source)  # [N,h,w]
    YY = torch.arange(0, h).view(1, h, 1).expand(N, h, w).type_as(img_source)

    grid_w = XX + disp * (ind_w_target - ind_w_source)
    grid_h = YY + disp * (ind_h_target - ind_h_source)

    grid_w_norm = 2.0 * grid_w / (w - 1) - 1.0
    grid_h_norm = 2.0 * grid_h / (h - 1) - 1.0

    grid = torch.stack((grid_w_norm, grid_h_norm), dim=3)  # [N,h,w,2]

    # img_source = torch.unsqueeze(img_source, 1)
    # print(img_source.shape)
    # print(grid.shape)
    # print(tt)
    img_target = F.grid_sample(img_source, grid, padding_mode='border', align_corners=False)  # [N,3,h,w]
    # img_target = torch.squeeze(img_target, 1)  # [N,h,w]
    return img_target
