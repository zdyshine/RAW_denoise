import argparse
import numpy as np
import torch
import torch.nn as nn
import os, rawpy, cv2
from tqdm import tqdm
from collections import OrderedDict
# from IQA_pytorch import SSIM
import dataloader
import logging
import random
from data_utils import *
import glob
####################################################
NET_NAME = 'GLResformer' # UNet1 | SCUNet | IDR
black_level = 1024
white_level = 16383
if not os.path.exists('./log/'):
    os.makedirs('./log/')

log_file_name = './log/' + NET_NAME + '_log.txt'
logging.basicConfig(level=logging.INFO, format='%(asctime)s  -  %(message)s', datefmt='%Y-%m-%d %H:%M:%S',
                    filename=log_file_name, filemode='a')
# set device
device = torch.device('cuda')

def get_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        print('param_group', param_group['lr'])
        param_group['lr'] = lr

def adjust_learning_rate(optimizer, epoch, step_size, lr_init, gamma):
    factor = epoch // step_size
    lr = lr_init * (gamma ** factor)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def load_network(load_path, network, strict=True):
    load_net = torch.load(load_path)
    load_net_clean = OrderedDict()  # remove unnecessary 'module.'
    for k, v in load_net.items():
        if k.startswith('module.'):
            load_net_clean[k[7:]] = v
        else:
            load_net_clean[k] = v
    network.load_state_dict(load_net_clean, strict=strict)

def main(args):
    logging.info(args)
    # set random seed
    print('========> Random Seed:', args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True
    # make dirs
    # save_path = os.path.join(args.save_path, NET_NAME)
    # os.makedirs(save_path, exist_ok=True)
    # set model NTIREv1
    print('=======>net name: ', NET_NAME)
    if NET_NAME == 'GLResformer':
        from network.network_gltormer import GLRestormer # \30 × 9
        net = GLRestormer()
    if args.pretrained:
        print('=========> Load From: ', args.pretrained)
        load_network(args.pretrained, net)
    print_network(net)
    net = nn.DataParallel(net)
    net.cuda()
    # set dataload
    train_data = dataloader.TraindatasetGL(args.dataroot, mode='train', patch_size=args.out_size, black_level= 1024, white_level=16383)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=32, pin_memory=True, drop_last=True)
    # val_data = dataloader.TraindatasetNpy(args.dataroot_val, mode='val', patch_size=512, black_level= 1024, white_level=16383)
    # val_loader = torch.utils.data.DataLoader(val_data, batch_size=1, shuffle=False, num_workers=2)
    val_noise_list = sorted(glob.glob(args.dataroot_val + '/*.dng'))[:2]
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr, weight_decay=0.0)
    loss_criterion = nn.L1Loss().cuda()

    iters = 0
    best_score = -1.0
    learning_rate = args.lr
    print('=========> Start Train ...')
    for epoch in range(1, args.epochs + 1):
        train_loss = 0
        train_loss_ssim = 0
        net.train()
        torch.set_grad_enabled(True)

        # for i, (LR_left, LR_right, HR_left, HR_right) in enumerate(tqdm(train_loader)):
        lr_decay_count = 0
        for i, (noise, clean, noisegl, clean_path, noise_path, noisegl_path) in enumerate(train_loader):
            if i < 5 and epoch == 1:
                print('clean_path', clean_path[0])
                print('noise_path', noise_path[0])
                print('noisegl_path', noisegl_path[0])
            iters += 1
            # adjust lr
            if iters % args.lr_decay_iters[lr_decay_count] == 0 and iters != 0:
                lr_decay_count +=1
                # adjust_learning_rate(optimizer, epoch, args.step_size, args.lr, args.gamma)
                # learning_rate = learning_rate / 5.0 # 2
                learning_rate = learning_rate / 2.0
                if learning_rate < args.min_lr:
                    learning_rate = args.min_lr
                get_learning_rate(optimizer, lr=learning_rate)

            clean, noise = clean.cuda(), noise.cuda()
            pred = net(noise, noisegl)

            ''' SR Loss '''
            loss_l1 = loss_criterion(pred, clean)
            loss = loss_l1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_loss_ssim += 0 # loss_s.item()
            if iters % 100 == 0:
                log_info = 'epoch:{}, Total iter:{}, iter:{}|{}, TLoss:{:.4f}, SLoss:{}. Lr:{}'.format(
                    epoch, iters, i+1, len(train_loader), train_loss / (i + 1), train_loss_ssim / (i + 1), learning_rate
                )
                print(log_info)
                logging.info(log_info)
            if loss > 10:
                log_info = 'Loss:{:.4f}'.format(loss.item())
                print(log_info)
                logging.info(log_info)
                exit()
            if iters % 5000 == 0:
                # val and save
                avg_score = validate(val_noise_list, net)
                is_best = avg_score > best_score
                best_score = max(avg_score, best_score)
                print('=====> save {} model.'.format(iters))
                save_checkpoint(epoch, net, avg_score)
                if is_best:
                    save_best_checkpoint(net)
                log_info = "===> Valid. score: {:.4f}, Best score: {:.4f}".format(avg_score, best_score)
                print(log_info)
                logging.info(log_info)
            if iters == args.total_iters:
                exit()
        # if epoch % 2 == 0:
        #     # val and save
        #     avg_score = validate(val_loader, net)
        #     is_best = avg_score > best_score
        #     best_score = max(avg_score, best_score)
        #     print('=====> save {} model.'.format(iters))
        #     save_checkpoint(epoch, net, avg_score)
        #     if is_best:
        #         save_best_checkpoint(net)
        #     log_info = "===> Valid. psnr: {:.4f}, Best psnr: {:.4f}".format(avg_score, best_score)
        #     print(log_info)
        #     logging.info(log_info)
        # if iters == args.total_iters:
        #     exit()

def validate(val_noise_list, net):
    net.eval()
    avg_score = 0
    for index, noise_path in enumerate(val_noise_list):
        print('noise_path', noise_path)
        raw_data_expand_c, height, width = read_image(noise_path)
        noise_gl = cv2.resize(raw_data_expand_c, (512, 512), interpolation=cv2.INTER_NEAREST)
        noise_gl = normalization(noise_gl, black_level, white_level)
        noise_gl = torch.from_numpy(
            np.ascontiguousarray(np.transpose(noise_gl, (2, 0, 1)))).float().unsqueeze(0)
        noise_gl = noise_gl.cuda()
        raw_data_expand_c_normal = normalization(raw_data_expand_c, black_level, white_level)
        raw_data_expand_c_normal = torch.from_numpy(np.transpose(
            raw_data_expand_c_normal.reshape(-1, height // 2, width // 2, 4), (0, 3, 1, 2))).float()
        raw_data_expand_c_normal = raw_data_expand_c_normal.cuda()
        pad = 106 #140
        m = nn.ReflectionPad2d(pad)
        imgs = m(raw_data_expand_c_normal)
        _, _, h, w = imgs.shape
        step = 300 #1000
        result_data = torch.zeros_like(imgs)
        for i in range(0, h, step):
            for j in range(0, w, step):
                if h - i < step + 2 * pad:
                    i = h - (step + 2 * pad)
                if w - j < step + 2 * pad:
                    j = w - (step + 2 * pad)
                clip = imgs[:, :, i:i + step + 2 * pad, j:j + step + 2 * pad]
                # clip = clip.cuda()
                with torch.no_grad():
                    g_images_clip = net(clip, noise_gl)
                result_data[:, :, i + pad:i + step + pad, j + pad:j + step + pad] = g_images_clip[:, :, pad:-pad, pad:-pad]
        result_data = result_data[:, :, pad:-pad, pad:-pad]

        result_data = result_data.cpu().detach().numpy().transpose(0, 2, 3, 1)
        result_data = inv_normalization(result_data, black_level, white_level)
        result_write_data = write_image(result_data, height, width)

        gt = rawpy.imread(noise_path.replace('noisy', 'ground_truth').replace('_noise', '_gt')).raw_image_visible

        pnsr_value = cal_psnr(gt, result_write_data)
        ssim_value = cal_ssim(gt, result_write_data)
        score = cal_score(pnsr_value, ssim_value)
        log_info = 'i:{}, pnsr_value:{:.3f}, ssim_value:{:.4f}, score:{:.4f}.'.format(index, pnsr_value, ssim_value, score)
        print(log_info)
        logging.info(log_info)
        avg_score += score
    net.train()
    return avg_score / len(val_noise_list)

# def validate(val_loader, net):
#     net.eval()
#     avg_score = 0
#     with torch.no_grad():
#         for i, (noise, clean, clean_path, noise_path) in enumerate(val_loader):
#             clean, noise = clean.cuda(), noise.cuda()
#             print('noise_path', noise_path[0])
#             # print(LR.shape)
#             result_data = net(noise)
#             result_data = result_data.cpu().detach().numpy().transpose(0, 2, 3, 1)
#             result_data = inv_normalization(result_data, black_level, white_level)
#             result_write_data = write_image(result_data, 1280, 1280)
#
#             clean = clean.cpu().detach().numpy().transpose(0, 2, 3, 1)
#             clean = inv_normalization(clean, black_level, white_level)
#             gt_write_data = write_image(clean, 1280, 1280)
#
#             pnsr_value = cal_psnr(gt_write_data, result_write_data)
#             ssim_value = cal_ssim(gt_write_data, result_write_data)
#             score = cal_score(pnsr_value, ssim_value)
#             # print('i:{}, pnsr_value:{:.3f}, ssim_value:{:.4f}, score:{:.4f}.'.format(i, pnsr_value, ssim_value, score))
#             log_info = 'i:{}, pnsr_value:{:.3f}, ssim_value:{:.4f}, score:{:.4f}.'.format(i, pnsr_value, ssim_value, score)
#             print(log_info)
#             logging.info(log_info)
#             avg_score += score
#     net.train()
#     return avg_score / len(val_loader)

def cal_score(psnr, ssim):
    # psnr和ssim为参赛者得到的psnr (dB)和ssim，psnr_min为psnr得分下限值，ssim_min为ssim得分下限值
    w, psnr_max, psnr_min, ssim_min = 0.8, 60, 30, 0.8
    Score = (w * max(psnr - psnr_min, 0) / (psnr_max - psnr_min) + (1 - w) * max(ssim - ssim_min, 0) / (
                1 - ssim_min)) * 100
    return Score

def save_checkpoint(iteration, model, psnr=1):
    model_folder = os.path.join('checkpoints', NET_NAME) #"model/PTMR/"
    os.makedirs(model_folder, exist_ok=True)
    model_out_path = model_folder + "/{}_{:06d}_{:.3f}.pth".format(NET_NAME, iteration, psnr)
    # model_out_path = model_folder + "/{}_{:06d}.pth".format(NET_NAME, iteration)

    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    torch.save(model.state_dict(), model_out_path)


def save_best_checkpoint(model):
    model_folder = os.path.join('checkpoints', NET_NAME)
    os.makedirs(model_folder, exist_ok=True)
    model_out_path = model_folder + "/best_model.pth"
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    torch.save(model.state_dict(), model_out_path)

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    # print(net)
    print('Total number of parameters: %d' % num_params)
    logging.info('Total number of parameters: %d' % num_params)

if __name__== '__main__':
    parser = argparse.ArgumentParser(description="ISR")
    # Model Selection
    parser.add_argument('--model', type=str, default='IMDN')
    # Hardware Setting
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--seed', type=int, default=202202, help='Random seed')
    parser.add_argument("--ext", type=str, default='.npy')

    # Directory Setting
    parser.add_argument('--dataroot', type=str, default='/codec/zhangdy/datasets/dataset/crop')
    parser.add_argument('--dataroot_val', type=str, default='/codec/zhangdy/datasets/dataset/noisy')
    parser.add_argument('--scale', type=int, default=1, help='train scale')
    parser.add_argument('--out_size', type=int, default=320, help='target size') # 128 | 144 | 160 | 96
    parser.add_argument("--pretrained", default="", type=str, help="path to pretrained models")

    # Learning Options
    parser.add_argument('--epochs', type=int, default=1000000, help='Max Epochs')
    parser.add_argument('--total_iters', type=int, default=1200000, help='Max Epochs')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size')
    parser.add_argument('--loss', type=str, default='l1', help='loss function configuration')

    # Optimization specifications
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate') # 1e-4
    parser.add_argument('--lr_decay_iters', type=list, default=[250000, 400000, 600000, 2000000], help='learning rate decay per N epochs')
    parser.add_argument('--min_lr', type=float, default=1e-7, help='min learning rate')
    parser.add_argument('--decay_type', type=str, default='step', help='learning rate decay type')
    parser.add_argument('--gamma', type=float, default=0.5, help='learning rate decay factor for step decay')
    parser.add_argument('--optimizer', default='ADAM', choices=('SGD', 'ADAM', 'RMSprop', 'ADAMax'),
                        help='optimizer to use (SGD | ADAM | RMSprop | ADAMax)')
    parser.add_argument('--weight_decay', type=float, default=0, help='weight decay')
    parser.add_argument("--debug", type=bool, default=False)

    args = parser.parse_args()
    print(args)
    main(args)

    # dataload_train = dataset_pairedsr.TrainDataset(args)
    # for i in range(len(dataload_train.ImagePath)):
    #     dataload_train.__getitem__(i)

    # dataload_val = dataset_pairedsr.ValDataset(args)
    # for i in range(len(dataload_val.ImagePath)):
    #     dataload_val.__getitem__(i)
    # cv2.destroyWindow()

