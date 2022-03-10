import argparse
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

import os
from os.path import join
from collections import defaultdict
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import math

from utils import dataset, util, loss_func
from model.model import DispNet
from model import net_utils
from utils.loss_func import Weighted_Rec_Loss, Masked_Quarter_Min_Loss, Edge_Aware_Smoothness_Loss, Conf_Prior_Loss
from model.net_utils import lf2sublfs

# Training settings
parser = argparse.ArgumentParser(description="LF depth estimation: train")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
parser.add_argument("--step", type=int, default=250, help="Learning rate decay every n epochs")
parser.add_argument("--reduce", type=float, default=0.5, help="Learning rate decay")
parser.add_argument("--patch_size", type=int, default=128, help="Training patch size")
parser.add_argument("--batch_size", type=int, default=4, help="Training batch size")
parser.add_argument("--resume_epoch", type=int, default=0, help="resume from checkpoint epoch")
parser.add_argument("--max_epoch", type=int, default=2000, help="maximum epoch for training")
parser.add_argument("--num_cp", type=int, default=100, help="Number of epochs for saving checkpoint")
parser.add_argument("--num_snapshot", type=int, default=1, help="Number of epochs for saving loss figure")

parser.add_argument("--dataset", type=str, default="HCI", help="Dataset for training")
parser.add_argument("--dataset_path", type=str, default="./LFData/train_HCI_12LF_RGB.h5", help="Dataset file for training")
parser.add_argument("--angular_num", type=int, default=7, help="angular number of the light field ")
parser.add_argument("--weight_smooth", type=float, default=0.1, help="weight for smooth loss ")
parser.add_argument("--weight_conf", type=float, default=0.0, help="weight for confidence loss ")
parser.add_argument("--loss_crop", type=int, default=8, help="crop the patch boundary when training")

parser.add_argument("--std_thres", type=float, default=0.2)

# parser.add_argument("--loss", type=str, default='MaskQuarterMinLoss')

opt = parser.parse_args()


# print(opt)


def main():
    print(opt)
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(1)

    # Data loader
    print('===> Loading datasets')
    train_set = dataset.TrainDataFromHdf5(opt)
    train_loader = DataLoader(dataset=train_set, batch_size=opt.batch_size, shuffle=True)
    print('loaded {} LFIs from {}'.format(len(train_loader), opt.dataset_path))

    # Build model
    print("===> building net")
    opt.sub_lf_num = math.ceil(opt.angular_num / 2) ** 2
    opt.ind_ref = (opt.angular_num - 1) // 2 * (opt.angular_num + 1)
    print('sub lf num', opt.sub_lf_num)
    print('ind ref', opt.ind_ref)
    model = DispNet(opt).to(device)



    # optimizer and loss logger
    print("===> setting optimizer")
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=opt.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=opt.step, gamma=opt.reduce)
    losslogger = defaultdict(list)

    l1loss = torch.nn.L1Loss()

    # model dir
    model_dir = 'Coarse_ckp_{}_an{}_sm{}_cf{}_ps{}_bs{}_lr{}'.format(
        opt.dataset, opt.angular_num,opt.weight_smooth, opt.weight_conf, opt.patch_size,opt.batch_size, opt.lr)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # optionally resume from a checkpoint
    if opt.resume_epoch:
        resume_path = join(model_dir, 'model_epoch_{}.pth'.format(opt.resume_epoch))
        if os.path.isfile(resume_path):
            print("==>loading checkpoint 'epoch{}'".format(resume_path))
            checkpoint = torch.load(resume_path)
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            losslogger = checkpoint['losslogger']
        else:
            print("==> no model found at 'epoch{}'".format(opt.resume_epoch))

    # training
    print("===> training")
    for epoch in range(opt.resume_epoch + 1, opt.max_epoch):
        model.train()
        loss_count = 0.
        for k in range(50):
            for t, batch in enumerate(train_loader, 1):
                # print(t)
                lf = batch[0].to(device)  # [N,an2,3,h,w]
                N, an2, c, h, w = lf.shape

                subLFs = lf2sublfs(lf)

                out_disp_coarse_sub, out_conf_coarse_sub, out_disp_coarse = model(subLFs)


                loss = 0.
                for s in range(len(out_disp_coarse)):
                    disp_coarse_sub = out_disp_coarse_sub[s]
                    conf_coarse_sub = out_conf_coarse_sub[s]
                    disp_coarse = out_disp_coarse[s]

                    loss_coarse_rec = Weighted_Rec_Loss(lf, disp_coarse_sub, conf_coarse_sub, opt)
                    loss_coarse_conf = Conf_Prior_Loss(conf_coarse_sub, opt.loss_crop)

                    loss_coarse_smooth = (
                            Edge_Aware_Smoothness_Loss(disp_coarse_sub[:,0:1], lf[:,opt.ind_ref])
                            + Edge_Aware_Smoothness_Loss(disp_coarse_sub[:,1:2], lf[:,opt.ind_ref])
                            + Edge_Aware_Smoothness_Loss(disp_coarse_sub[:,2:3], lf[:,opt.ind_ref])
                            + Edge_Aware_Smoothness_Loss(disp_coarse_sub[:,3:4], lf[:,opt.ind_ref]))/4.

                    loss += loss_coarse_rec + opt.weight_conf * loss_coarse_conf + opt.weight_smooth*loss_coarse_smooth

                    lf = F.interpolate(lf.reshape(N,an2*c,h,w), scale_factor=1/2., mode='bicubic', align_corners=False)
                    h = h//2
                    w = w//2
                    lf = lf.reshape(N, an2, c, h, w)
                    # print(lf.shape)


                loss = loss/len(out_disp_coarse)

                loss_count += loss.item()


                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        scheduler.step()

        losslogger['epoch'].append(epoch)
        losslogger['loss'].append(loss_count / len(train_loader))

        # checkpoint
        if epoch % opt.num_cp == 0:
            model_save_path = join(model_dir, "model_epoch_{}.pth".format(epoch))
            state = {'epoch': epoch, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                     'scheduler': scheduler.state_dict(), 'losslogger': losslogger, }
            torch.save(state, model_save_path)
            print("checkpoint saved to {}".format(model_save_path))

        # loss snapshot
        if epoch % opt.num_snapshot == 0:
            plt.figure()
            plt.title('loss')
            plt.plot(losslogger['epoch'], losslogger['loss'])
            plt.savefig(model_dir + ".jpg")
            plt.close()


if __name__ == "__main__":
    main()





