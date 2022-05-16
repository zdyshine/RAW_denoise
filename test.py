import glob, os
import time
import cv2
import argparse
from collections import OrderedDict
import torch.nn as nn
import torch
import numpy as np

from data_utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--numOfWorkers', type=int, default=0, help='workers for dataloader')
parser.add_argument('--batchSize', type=int, default=1)
parser.add_argument('--loadSize', type=int, default=512, help='image loading size')
# parser.add_argument('--dataRoot', type=str, default=r'/test/zhangdy/dataset/denoise/test_data')
parser.add_argument('--dataRoot', type=str, default=r'D:\zdy\3_interest\2022_denoise\dataset\testset')
parser.add_argument('--dataRootVal', type=str, default=r'D:\zdy\3_interest\2022_denoise\dataset\noisy')
# parser.add_argument('--pretrained', type=str, default='./checkpoints/CasmerMix/CasmerMix_000003_76.820.pth', help='pretrained models for finetuning') #
# parser.add_argument('--pretrained', type=str, default='./checkpoints/interp_Casmer_05.pth', help='pretrained models for finetuning')
parser.add_argument('--pretrained', type=str, default=r'D:\zdy\3_interest\2022_denoise\checkpoints\CasmerMix_000002_76.941.pth', help='pretrained models for finetuning')
# parser.add_argument('--pretrained', type=str, default=r'./checkpoints/interp_Casmer_05.pth', help='pretrained models for finetuning')
# parser.add_argument('--savePath', type=str, default='./results/unet/')
# parser.add_argument('--savePath', type=str, default=r'D:\zdy\3_interest\2022_denoise\dataset\results\NAF')
# parser.add_argument('--savePath', type=str, default=r'./results/data')
parser.add_argument('--savePath', type=str, default=r'D:\zdy\3_interest\2022_denoise\submit\result\data')
parser.add_argument('--net', type=str, default='Casmer')
parser.add_argument('--black_level', type=int, default=1024)
parser.add_argument('--white_level', type=int, default=16383)
args = parser.parse_args()

def load_network(load_path, network, strict=True):
    load_net = torch.load(load_path)
    load_net_clean = OrderedDict()  # remove unnecessary 'module.'
    for k, v in load_net.items():
        if k.startswith('module.'):
            load_net_clean[k[7:]] = v
        else:
            load_net_clean[k] = v
    network.load_state_dict(load_net_clean, strict=strict)

batchSize = args.batchSize
loadSize = (args.loadSize, args.loadSize)
dataRoot = args.dataRoot
savePath = args.savePath
black_level = args.black_level
white_level = args.white_level

if not os.path.exists(savePath):
    os.makedirs(savePath)

if args.net == 'unet':
    from network.network_unet import Unet  # \30 × 9
    net = Unet().cuda()  # 37M
elif args.net == 'IDR':
    from network.network_idr import IDR # \30 × 9
    net = IDR(num_c=96)

elif args.net == 'SCUNet':
    from network.network_scunet import SCUNet
    net = SCUNet(in_nc=4, config=[2, 2, 2, 2, 4, 2, 2], dim=64)

elif args.net == 'NAF':
    from network.network_naf import NAFNet  # \30 × 9
    net = NAFNet(img_channel=4, width=32, middle_blk_num=2,
                 enc_blk_nums=[2, 2, 2, 12], dec_blk_nums=[2, 2, 2, 2])
    '''
    NAF_000001_75.226, 56.53
    NAF_000017_75.799, 55.31
    NAF_000006_75.514, 54.21
    NAF_000016_75.696, 55.94
    NAF_000003_75.341, 56.53
    NAF_000004_75.319, 56.49
    NAF_000006_75.472, 56.25
    NAF_000007_75.358, 56.57        , top2
    NAF_000029_75.880, 55.05
    NAF_000031_75.589, 56.38
    NAF_000028_75.902, 55.66, 233,311
    NAF_000026_75.612, 55.37, 236,859
    NAF_000022_75.555, 56.21, 236,788
    NAF_000020_75.636, 56.68, 237,271, top1
    NAF_000019_75.737, 56.15, 236,904,
    NAF_000019_75.563, 56.15, 236,904,
    NAF_000019_75.563, 54.99, 237,314
    NAF_000016_75.696, 55.94, 236,988
    '''
elif args.net == 'NAFv1':
    from network.network_naf import NAFNet  # \30 × 9
    net = NAFNet(img_channel=4, width=32, middle_blk_num=2,
                 enc_blk_nums=[2, 2, 2, 4], dec_blk_nums=[2, 2, 2, 2])
    ''' 
    NAFv1_000005_74.514, 54.47
    NAFv1_000018_75.122, 56.28
    NAFv1_000023_75.201, 56.44
    NAFv1_000033_75.221, 56.66
    '''
elif args.net == 'Resformer':
    from network.network_restormer import Restormer  # \30 × 9
    # net = Restormer()
    # net = Restormer()
    net = Restormer(num_blocks=[2, 3, 3, 4]).cuda()
    ''' 
    Resformer_000003_76.067.pth, 58.05
    Resformer_000007_76.054,     58.07 # fusion 58.11 # 300+106
    '''
elif args.net == 'Casmer':
    from network.network_casmer import Casmer # \30 × 9
    net = Casmer()
    '''
    Casmer_000022_76.494, 58.56, L1
    Casmer_000002_76.450, 58.53, PSNR
    fusion: 58.72
    Casmer_000031_76.551, 58.88, L1
    Casmer_000032_76.583, 58.81, L1
    Casmer_000032_76.655, 58.87, L1
    Casmer_000034_76.661, 58.89, L1
    Casmer_000040_76.703, 58.92, L1
    Casmer_000044_76.715, 58.87, L1
    Casmer_000049_76.691, 58.88, L1
    Casmer_000009_76.555, 58.44, PSNR
    Casmer_000014_76.531, 58.46, PSNR
    Casmer_000006_76.497, 58.62, PSNR
    Casmer_000006_76.695, 59.06, PSNR
    Casmer_000011_76.740, 58.99, PSNR
    Casmer_000009_76.718, 59.07, PSNR
    Casmer_000016_76.702, 59.00, PSNR
    fusion: Casmer_000009_76.718 + Casmer_000040_76.703 = 59.15
    CasmerMix_000006_76.760, 58.94, L1
    CasmerMix_000006_76.690, 58.95, L1
    CasmerMix_000011_76.807, 58.95, L1
    CasmerMix_000002_76.712, 59.08, PSNR
    CasmerMix_000003_76.667, 59.13, PSNR * 256, ensemble: 59.41
    CasmerMix_000003_76.729, 59.09, PSNR *
    CasmerMix_000001_76.681, 59.13, PSNR, 320
    fusion: CasmerMix_000011_76.807 + CasmerMix_000003_76.667 = 59.23, ensemblex4: 59.45, ensemblex8: 59.47
    ######################################################################################################
    CasmerMix_000038_76.890, 59.03, PSNR, audodl, 256, 20, 24...
    CasmerMix_000008_76.876, 59.07, PSNR, audodl, 256, 20, 233270
    CasmerMix_000008_76.876, 59.07, PSNR, audodl, 256, 20, 233270
    CasmerMix_000040_76.845, 59.07, PSNR, audodl, 256, 20, 234537
    
    CasmerMix_000004_76.935, 59.14, PSNR, audodl, 384, 8,  236818
    CasmerMix_000007_76.915, ++++, PSNR, audodl, 384, 8,  ~~~~~
    CasmerMix_000010_76.956, 59.03, PSNR, audodl, 384, 8,  233943
    
    
    CasmerMix_000002_76.804, 59.12, L1,   3090,   320, 2,  233713
    
    CasmerMix_000019_76.838, 58.87, L1,   2080ti, 256, 5,  233713
    
    bash 
    '''
elif args.net == 'WeightCasmer':
    from network.network_caswmer import Caswmer  # \30 × 9
    net = Caswmer()
print('=========> Load From: ', args.pretrained)
load_network(args.pretrained, net)

for param in net.parameters():
    param.requires_grad = False

# print(noise_list)
print('OK!')
##################  val
# noise_list = glob.glob(args.dataRootVal + '/*.dng')[:2]
# from train import validate
# net.cuda()
# avg_score = validate(noise_list, net)
# print(avg_score) # 72.08784812915346  73.87825707606902  74.06188325564028

################### test
noise_list = glob.glob(dataRoot + '/*.dng')
print(noise_list)
TIME = []
net.eval()
net.cuda()
for index, input_path in enumerate(noise_list):
    img_name = os.path.basename(input_path)
    raw_data_expand_c, height, width = read_image(input_path)
    raw_data_expand_c_normal = normalization(raw_data_expand_c, black_level, white_level)
    raw_data_expand_c_normal = torch.from_numpy(np.transpose(
        raw_data_expand_c_normal.reshape(-1, height//2, width//2, 4), (0, 3, 1, 2))).float()

    pad = 64 # 96 # 64 # 106
    m = nn.ReflectionPad2d(pad)
    imgs = m(raw_data_expand_c_normal)
    print(index, raw_data_expand_c_normal.shape, raw_data_expand_c_normal.max(), raw_data_expand_c_normal.min())
    _, _, h, w = imgs.shape
    # rh, rw = h, w
    step = 128 # 64 # 128 # 300
    res = torch.zeros_like(imgs)
    for i in range(0, h, step):
        for j in range(0, w, step):
            if h - i < step + 2 * pad:
                i = h - (step + 2 * pad)
            if w - j < step + 2 * pad:
                j = w - (step + 2 * pad)
            clip = imgs[:, :, i:i + step + 2 * pad, j:j + step + 2 * pad]
            # print(clip.shape)
            clip = clip.cuda()
            start = time.time()
            with torch.no_grad():
                g_images_clip = net(clip)
                # g_images_clip += (net(clip.transpose(2, 3).flip(3))).flip(3).transpose(2, 3)  # z90
                # g_images_clip += (net(clip.flip(2).flip(3))).flip(2).flip(3)  # z180
                # g_images_clip += (net(clip.transpose(2, 3).flip(2))).flip(2).transpose(2, 3)  # z270

                # g_images_clip += (net(clip.flip(3))).flip(3)  # f0
                # g_images_clip += (net(clip.transpose(2, 3))).transpose(2, 3)  # f90
                # g_images_clip += (net(clip.flip(2))).flip(2)  # f180
                # g_images_clip += (net(clip.flip(3).transpose(2, 3).flip(2))).flip(2).transpose(2, 3).flip(3)  # f270
                # g_images_clip = g_images_clip / 2
            res[:, :, i + pad:i + step + pad, j + pad:j + step + pad] = g_images_clip[:, :, pad:-pad, pad:-pad]
    res = res[:, :, pad:-pad, pad:-pad]
    # print(res.shape)
    TIME.append(time.time() - start)

    """
    post-process
    """
    result_data = res.cpu().detach().numpy().transpose(0, 2, 3, 1)
    result_data = inv_normalization(result_data, black_level, white_level)
    result_write_data = write_image(result_data, height, width)
    output_path = os.path.join(savePath, img_name.replace('noisy', 'denoise'))
    # print(output_path)
    write_back_dng(input_path, output_path, result_write_data)

print('total time: {}, avg_time: {}.'.format(np.sum(TIME), np.mean(TIME)))
