
import torch.utils.data as data
import torch
import h5py
import numpy as np
import random
import math
import os
from scipy import io

class TrainDataFromHdf5(data.Dataset):
    def __init__(self, opt):
        super(TrainDataFromHdf5, self).__init__()

        hf = h5py.File(opt.dataset_path, 'r')
        self.img_HR = hf.get('img_HR')  # [N,ah,aw,c,h,w]

        self.psize = opt.patch_size
        self.angular_num = opt.angular_num

    def __getitem__(self, index):

        # get one item
        img = self.img_HR[index]  # [ah,aw,c,h,w]
        # print('index', index)

        # io.savemat('img_full.mat', {"img": img[4,4]})

        an = img.shape[0]
        H = img.shape[3]
        W = img.shape[4]

        x = random.randrange(0, H - self.psize, 8)
        y = random.randrange(0, W - self.psize, 8)

        an_crop = math.ceil((an-self.angular_num)/2)
        img = img[an_crop:an_crop+self.angular_num, an_crop:an_crop+self.angular_num, :, x:x + self.psize, y:y + self.psize]  # [ah,aw,ph,pw]

        # 4D augmentation
        # flip
        if np.random.rand(1) > 0.5:
            img = np.flip(np.flip(img, 0), 3)

        if np.random.rand(1) > 0.5:
            img = np.flip(np.flip(img, 1), 4)

        # rotate
        r_ang = np.random.randint(1, 5)
        img = np.rot90(img, r_ang, (3, 4))
        img = np.rot90(img, r_ang, (0, 1))

        # io.savemat('img_patch.mat', {"patch": img[4, 4]})
        # to tensor
        img = img.reshape(-1, 3, self.psize, self.psize)  # [an,3,ph,pw]
        img = torch.from_numpy(img.astype(np.float32) / 255.0)

        return [img]

    def __len__(self):
        return self.img_HR.shape[0]


class TestDataFromHdf5(data.Dataset):
    def __init__(self, opt):
        super(TestDataFromHdf5, self).__init__()

        files = os.listdir(opt.test_path)

        self.test_data = []
        self.test_name = []
        for file in files:
            f = opt.test_path + "/" + file
            data = io.loadmat(f)
            self.test_data.append(data['LF'])
            self.test_name.append(file[:-4])

        # hf = h5py.File(opt.dataset_path, 'r')
        # self.img_HR = hf.get('img_HR')  # [N,ah,aw,c,h,w]

        self.angular_num = opt.angular_num

    def __getitem__(self, index):

        # get one item
        # img = self.img_HR[index, :self.an, :self.an]  # [ah,aw,c,h,w]
        img = self.test_data[index]
        img_name = self.test_name[index]
        # print(img.shape)
        # print(img_name)

        an, _, c, h, w = img.shape
        an_crop = math.ceil((an-self.angular_num)/2)
        img = img[an_crop:an_crop+self.angular_num, an_crop:an_crop+self.angular_num]  # [ah,aw,c,ph,pw]
        img = img.reshape(-1, c, h, w)
        # print(img.shape)

        # to tensor
        img = torch.from_numpy(img.astype(np.float32) / 255.0)
        return img, img_name

    def __len__(self):
        return len(self.test_data)