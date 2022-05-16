import sys
import torch
from collections import OrderedDict

# alpha = float(sys.argv[1])
alpha = float(0.5)

net_ESRGAN_path = r'D:\zdy\3_interest\2022_denoise\checkpoints\CasmerMix_000004_76.935.pth'# 59.14
net_PSNR_path = './checkpoints/CasmerMix/CasmerMix_000002_76.804.pth'# 59.13

net_interp_path = r'D:\net_final.pth'.format(int(alpha*10))

net_PSNR = torch.load(net_PSNR_path)
net_ESRGAN = torch.load(net_ESRGAN_path)
net_interp = OrderedDict()

print('Interpolating with alpha = ', alpha)

for k, v_PSNR in net_PSNR.items():
    v_ESRGAN = net_ESRGAN[k]
    net_interp[k] = (1 - alpha) * v_PSNR + alpha * v_ESRGAN

torch.save(net_interp, net_interp_path)