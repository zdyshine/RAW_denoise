import random
import numpy as np
from torch.utils.data import Dataset
import torch
import glob
import os
import rawpy
# from matplotlib import pyplot as plt

class Mixing_Augment:
    def __init__(self, mixup_beta=1.2, use_identity=True):
        self.dist = torch.distributions.beta.Beta(torch.tensor([mixup_beta]), torch.tensor([mixup_beta]))

        self.use_identity = use_identity

        self.augments = [self.mixup]

    def mixup(self, target, input_):
        lam = self.dist.rsample((1, 1)).item()

        r_index = torch.randperm(target.size(0)).cuda()

        target = lam * target + (1 - lam) * target[r_index, :]
        input_ = lam * input_ + (1 - lam) * input_[r_index, :]

        return target, input_

    def __call__(self, target, input_):
        if self.use_identity:
            augment = random.randint(0, len(self.augments))
            if augment < len(self.augments):
                target, input_ = self.augments[augment](target, input_)
        else:
            augment = random.randint(0, len(self.augments) - 1)
            target, input_ = self.augments[augment](target, input_)
        return target, input_

def normalization(input_data, black_level, white_level):
    output_data = (input_data.astype(float) - black_level) / (white_level - black_level)
    # output_data = np.clip(output_data, 0, 1)
    return output_data

def read_image(input_path, black_level, white_level):
    raw = rawpy.imread(input_path)
    raw_data = raw.raw_image_visible
    height = raw_data.shape[0]
    width = raw_data.shape[1]

    raw_data_expand = np.expand_dims(raw_data, axis=2)
    raw_data_expand_c = np.concatenate((raw_data_expand[0:height:2, 0:width:2, :],
                                        raw_data_expand[0:height:2, 1:width:2, :],
                                        raw_data_expand[1:height:2, 0:width:2, :],
                                        raw_data_expand[1:height:2, 1:width:2, :]), axis=2)
    # print(raw_data_expand_c.max(), raw_data_expand_c.min())
    # raw_data_expand_c.shape: (H,W,4)
    raw_data_expand_c = normalization(raw_data_expand_c, black_level, white_level)
    return raw_data_expand_c #, height, width

def npy_loader(path):
    return np.load(path)

def augment_img(img, mode=0):
    if mode == 0:
        return img
    elif mode == 1:
        return np.flipud(np.rot90(img))
    elif mode == 2:
        return np.flipud(img)
    elif mode == 3:
        return np.rot90(img, k=3)
    elif mode == 4:
        return np.flipud(np.rot90(img, k=2))
    elif mode == 5:
        return np.rot90(img)
    elif mode == 6:
        return np.rot90(img, k=2)
    elif mode == 7:
        return np.flipud(np.rot90(img, k=3))

class Traindataset(Dataset):
    def __init__(self, dataroot, mode='train', patch_size=512, black_level= 1024, white_level=16383):
        super(Traindataset, self).__init__()
        # print(os.path.join(opt.train_root, 'gt', '*.png'))
        paths_clean = sorted(glob.glob(os.path.join(dataroot, 'ground_truth', '*.dng')))
        paths_noise = sorted(glob.glob(os.path.join(dataroot, 'noisy', '*.dng')))
        # print('1111111111111111',len(paths_noise), len(paths_clean))
        if mode == 'train':
            self.paths_clean = paths_clean[:-2] * 10
            self.paths_noise = paths_noise[:-2] * 10
        elif mode == 'val':
            self.paths_clean = paths_clean[-2:]
            self.paths_noise = paths_noise[-2:]
        else:
            print('mode is train or val')

        print('=========> Total train/val images: {}.'.format(len(self.paths_clean)))
        assert self.paths_clean, 'Error: GT path is empty.'
        # if self.paths_LQ and self.paths_GT:
        assert len(self.paths_noise) == len(self.paths_clean), 'GT and LQ datasets have different number of images - {}, {}.'.format(
            len(self.paths_noise), len(self.paths_clean))
        # self.opt = opt
        self.mode = mode
        self.ps = patch_size # opt.out_size
        self.black_level = black_level
        self.white_level = white_level

    def __getitem__(self, index):
        GT_path, LQ_path = None, None

        # get  image
        clean_path = self.paths_clean[index]
        noise_path = self.paths_clean[index].replace('ground_truth', 'noisy').replace('gt', 'noise')
        # img_clean, h_c, w_c = read_image(clean_path, self.black_level, self.white_level) # (H,W,4)
        # img_noise, h_n, w_n = read_image(noise_path, self.black_level, self.white_level) # (H,W,4)
        img_clean = read_image(clean_path, self.black_level, self.white_level) # (H,W,4)
        img_noise = read_image(noise_path, self.black_level, self.white_level) # (H,W,4)

        # print('111',img_noise.shape, img_clean.shape)
        Hc, Wc, C = img_clean.shape
        Hn, Wn, C = img_noise.shape
        if Hc != Hn or Wc != Wn:
            print('*******wrong image*******:{}'.format(noise_path))
        if self.mode == 'train':
            # --------------------------------
            # randomly crop
            # --------------------------------
            rnd_h = random.randint(0, max(0, Hc - self.ps))
            rnd_w = random.randint(0, max(0, Wc - self.ps))

            img_noise = img_noise[rnd_h:rnd_h + self.ps, rnd_w:rnd_w + self.ps, :]
            img_clean = img_clean[rnd_h:rnd_h + self.ps, rnd_w:rnd_w + self.ps, :]

            # # --------------------------------
            # # augmentation - flip and/or rotate
            # # --------------------------------
            # if np.random.randint(2, size=1)[0] == 1:  # random flip
            #     img_noise = np.flip(img_noise, axis=1)  # H
            #     img_clean = np.flip(img_clean, axis=1)
            # if np.random.randint(2, size=1)[0] == 1:
            #     img_noise = np.flip(img_noise, axis=2)  # W
            #     img_clean = np.flip(img_clean, axis=2)
            # if np.random.randint(2, size=1)[0] == 1:  # random transpose
            #     img_noise = np.transpose(img_noise, (1, 0, 2)) # 0,1,2
            #     img_clean = np.transpose(img_clean, (1, 0, 2))

            mode = random.randint(0, 7)
            img_noise, img_clean = augment_img(img_noise, mode=mode), augment_img(img_clean, mode=mode)

        if self.mode == 'val':
            # --------------------------------
            # crop
            # --------------------------------
            rnd_h = int(Hc // 4)
            rnd_w = int(Wc // 4)
            img_noise = img_noise[rnd_h:rnd_h + 640, rnd_w:rnd_w + 640, :]
            img_clean = img_clean[rnd_h:rnd_h + 640, rnd_w:rnd_w + 640, :]

        # print('333',img_noise.shape, img_clean.shape)

        # HWC to CHW, numpy to tensor
        img_noise = torch.from_numpy(
            np.ascontiguousarray(np.transpose(img_noise, (2, 0, 1)))).float()
        img_clean = torch.from_numpy(
            np.ascontiguousarray(np.transpose(img_clean, (2, 0, 1)))).float()
        # print(img_LQ.shape, GT_path.replace('HR', 'LR'), img_GT.shape, GT_path)
        return img_noise, img_clean  # , LQ_path, GT_path

    def __len__(self):
        return len(self.paths_clean)

class TraindatasetNpy(Dataset):
    def __init__(self, dataroot, mode='train', patch_size=512, black_level= 1024, white_level=16383):
        super(TraindatasetNpy, self).__init__()
        # print(os.path.join(opt.train_root, 'gt', '*.png'))
        paths_clean = sorted(glob.glob(os.path.join(dataroot, 'ground_truth_crop/*', '*.npy')))
        paths_noise = sorted(glob.glob(os.path.join(dataroot, 'noisy_crop/*', '*.npy')))
        # print('1111111111111111',len(paths_noise), len(paths_clean))
        if mode == 'train':
            self.paths_clean = paths_clean #[:-2] * 10
            self.paths_noise = paths_noise #[:-2] * 10
        elif mode == 'val':
            self.paths_clean = paths_clean #[-2:]
            self.paths_noise = paths_noise #[-2:]
        else:
            print('mode is train or val')

        print('=========> Total train/val images: {}.'.format(len(self.paths_clean)))
        assert self.paths_clean, 'Error: GT path is empty.'
        # if self.paths_LQ and self.paths_GT:
        assert len(self.paths_noise) == len(self.paths_clean), 'GT and LQ datasets have different number of images - {}, {}.'.format(
            len(self.paths_noise), len(self.paths_clean))
        # self.opt = opt
        self.mode = mode
        self.ps = patch_size # opt.out_size
        self.black_level = black_level
        self.white_level = white_level

    def __getitem__(self, index):
        GT_path, LQ_path = None, None

        # get  image
        clean_path = self.paths_clean[index]
        noise_path = self.paths_clean[index].replace('ground_truth_crop', 'noisy_crop').replace('gt', 'noise')
        # print('clean_path', clean_path)
        # print('noise_path', noise_path)
        # exit()
        img_clean = npy_loader(clean_path) # (H,W,4)
        img_clean = normalization(img_clean, self.black_level, self.white_level)

        img_noise = npy_loader(noise_path) # (H,W,4)
        img_noise = normalization(img_noise, self.black_level, self.white_level)

        # print('111',img_noise.shape, img_clean.shape)
        Hc, Wc, C = img_clean.shape
        Hn, Wn, C = img_noise.shape
        if Hc != Hn or Wc != Wn:
            print('*******wrong image*******:{}'.format(noise_path))
        if self.mode == 'train':
            # --------------------------------
            # randomly crop
            # --------------------------------
            rnd_h = random.randint(0, max(0, Hc - self.ps))
            rnd_w = random.randint(0, max(0, Wc - self.ps))

            img_noise = img_noise[rnd_h:rnd_h + self.ps, rnd_w:rnd_w + self.ps, :]
            img_clean = img_clean[rnd_h:rnd_h + self.ps, rnd_w:rnd_w + self.ps, :]

            mode = random.randint(0, 7)
            img_noise, img_clean = augment_img(img_noise, mode=mode), augment_img(img_clean, mode=mode)

        # HWC to CHW, numpy to tensor
        img_noise = torch.from_numpy(
            np.ascontiguousarray(np.transpose(img_noise, (2, 0, 1)))).float()
        img_clean = torch.from_numpy(
            np.ascontiguousarray(np.transpose(img_clean, (2, 0, 1)))).float()
        # print(img_LQ.shape, GT_path.replace('HR', 'LR'), img_GT.shape, GT_path)
        return img_noise, img_clean, clean_path, noise_path

    def __len__(self):
        return len(self.paths_clean)

class TraindatasetGL(Dataset):
    def __init__(self, dataroot, mode='train', patch_size=512, black_level= 1024, white_level=16383):
        super(TraindatasetGL, self).__init__()
        # print(os.path.join(opt.train_root, 'gt', '*.png'))
        paths_clean = sorted(glob.glob(os.path.join(dataroot, 'ground_truth_crop/*', '*.npy')))
        paths_noise = sorted(glob.glob(os.path.join(dataroot, 'noisy_crop/*', '*.npy')))
        # print('1111111111111111',len(paths_noise), len(paths_clean))
        if mode == 'train':
            self.paths_clean = paths_clean #[:-2] * 10
            self.paths_noise = paths_noise #[:-2] * 10
        elif mode == 'val':
            self.paths_clean = paths_clean #[-2:]
            self.paths_noise = paths_noise #[-2:]
        else:
            print('mode is train or val')

        print('=========> Total train/val images: {}.'.format(len(self.paths_clean)))
        assert self.paths_clean, 'Error: GT path is empty.'
        # if self.paths_LQ and self.paths_GT:
        assert len(self.paths_noise) == len(self.paths_clean), 'GT and LQ datasets have different number of images - {}, {}.'.format(
            len(self.paths_noise), len(self.paths_clean))
        # self.opt = opt
        self.mode = mode
        self.ps = patch_size # opt.out_size
        self.black_level = black_level
        self.white_level = white_level
        self.paths_gl = '/codec/zhangdy/datasets/dataset/noisy'
        #self.paths_gl = os.path.join(dataroot.replace('crop', 'gl_npy'), 'noisy_npy')

    def __getitem__(self, index):
        GT_path, LQ_path = None, None

        # get  image
        clean_path = self.paths_clean[index]
        noise_path = self.paths_clean[index].replace('ground_truth_crop', 'noisy_crop').replace('gt', 'noise')
        ext = noise_path.split('/')[7]
        #noisegl_path = os.path.join(self.paths_gl, ext) +  '.npy'
        noisegl_path = os.path.join(self.paths_gl, ext + '_noise.npy')
        #print('clean_path', clean_path)
        #print('noise_path', noise_path)
        #print('noisegl_path', noisegl_path)
        #exit()
        img_clean = npy_loader(clean_path) # (H,W,4)
        img_clean = normalization(img_clean, self.black_level, self.white_level)

        img_noise = npy_loader(noise_path) # (H,W,4)
        img_noise = normalization(img_noise, self.black_level, self.white_level)

        noisegl = npy_loader(noisegl_path) # (H,W,4)
        noisegl = normalization(noisegl, self.black_level, self.white_level)


        # print('111',img_noise.shape, img_clean.shape)
        Hc, Wc, C = img_clean.shape
        Hn, Wn, C = img_noise.shape
        if Hc != Hn or Wc != Wn:
            print('*******wrong image*******:{}'.format(noise_path))
        if self.mode == 'train':
            # --------------------------------
            # randomly crop
            # --------------------------------
            rnd_h = random.randint(0, max(0, Hc - self.ps))
            rnd_w = random.randint(0, max(0, Wc - self.ps))

            img_noise = img_noise[rnd_h:rnd_h + self.ps, rnd_w:rnd_w + self.ps, :]
            img_clean = img_clean[rnd_h:rnd_h + self.ps, rnd_w:rnd_w + self.ps, :]

            mode = random.randint(0, 7)
            img_noise, img_clean = augment_img(img_noise, mode=mode), augment_img(img_clean, mode=mode)
            noisegl = augment_img(noisegl, mode=mode)


        # HWC to CHW, numpy to tensor
        img_noise = torch.from_numpy(
            np.ascontiguousarray(np.transpose(img_noise, (2, 0, 1)))).float()
        img_clean = torch.from_numpy(
            np.ascontiguousarray(np.transpose(img_clean, (2, 0, 1)))).float()
        noisegl = torch.from_numpy(
            np.ascontiguousarray(np.transpose(noisegl, (2, 0, 1)))).float()
        # print(img_LQ.shape, GT_path.replace('HR', 'LR'), img_GT.shape, GT_path)
        return img_noise, img_clean, noisegl, clean_path, noise_path, noisegl_path

    def __len__(self):
        return len(self.paths_clean)


if __name__ == '__main__':
    from data_utils import inv_normalization, write_image, write_back_dng
    dataroot = r'D:\zdy\3_interest\2022_denoise\dataset\crop'
    test_data = TraindatasetNpy(dataroot, mode='val', patch_size=512, black_level= 1024, white_level=16383)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=True, num_workers=1)
    for i, (img_noise, img_clean) in enumerate(test_loader):
        print(img_clean.shape, img_noise.shape)

        # result_data = img_noise.cpu().detach().numpy().transpose(0, 2, 3, 1)
        # # result_data = inv_normalization(result_data, 1024, 16383)
        # result_write_data = write_image(result_data, 1280, 1280)
        # write_back_dng(r'D:\zdy\3_interest\2022_denoise\dataset\noisy\{}_noise.dng'.format(i), r'D:\zdy\3_interest\2022_denoise\dataset\{}_noise.dng'.format(i), result_write_data)
        #
        # clean = img_clean.cpu().detach().numpy().transpose(0, 2, 3, 1)
        # # clean = inv_normalization(clean, 1024, 16383)
        # gt_write_data = write_image(clean, 1280, 1280)
        # write_back_dng(r'D:\zdy\3_interest\2022_denoise\dataset\ground_truth\{}_gt.dng'.format(i), r'D:\zdy\3_interest\2022_denoise\dataset\{}_noise.dng'.format(i), result_write_data)
        #
