import os
import numpy as np
import rawpy
import torch
import skimage.metrics
from matplotlib import pyplot as plt
# from unetTorch import Unet
import argparse


def normalization(input_data, black_level, white_level):
    output_data = (input_data.astype(float) - black_level) / (white_level - black_level)
    return output_data


def inv_normalization(input_data, black_level, white_level):
    output_data = np.clip(input_data, 0., 1.) * (white_level - black_level) + black_level
    output_data = output_data.astype(np.uint16)
    return output_data


def write_back_dng(src_path, dest_path, raw_data):
    """
    replace dng data
    """
    width = raw_data.shape[0]
    height = raw_data.shape[1]
    falsie = os.path.getsize(src_path)
    data_len = width * height * 2
    header_len = 8

    with open(src_path, "rb") as f_in:
        data_all = f_in.read(falsie)
        dng_format = data_all[5] + data_all[6] + data_all[7]

    with open(src_path, "rb") as f_in:
        header = f_in.read(header_len)
        if dng_format != 0:
            _ = f_in.read(data_len)
            meta = f_in.read(falsie - header_len - data_len)
        else:
            meta = f_in.read(falsie - header_len - data_len)
            _ = f_in.read(data_len)

        data = raw_data.tobytes()

    with open(dest_path, "wb") as f_out:
        f_out.write(header)
        if dng_format != 0:
            f_out.write(data)
            f_out.write(meta)
        else:
            f_out.write(meta)
            f_out.write(data)

    if os.path.getsize(src_path) != os.path.getsize(dest_path):
        print("replace raw data failed, file size mismatch!")
    else:
        print("replace raw data finished")


def read_image(input_path):
    raw = rawpy.imread(input_path)
    # r = raw.raw_image
    raw_data = raw.raw_image_visible
    height = raw_data.shape[0]
    width = raw_data.shape[1]

    raw_data_expand = np.expand_dims(raw_data, axis=2)
    raw_data_expand_c = np.concatenate((raw_data_expand[0:height:2, 0:width:2, :],
                                        raw_data_expand[0:height:2, 1:width:2, :],
                                        raw_data_expand[1:height:2, 0:width:2, :],
                                        raw_data_expand[1:height:2, 1:width:2, :]), axis=2)
    # raw_data_expand_c.shape: (H,W,4)
    return raw_data_expand_c, height, width


def write_image(input_data, height, width):
    output_data = np.zeros((height, width), dtype=np.uint16)
    for channel_y in range(2):
        for channel_x in range(2):
            output_data[channel_y:height:2, channel_x:width:2] = input_data[0:, :, :, 2 * channel_y + channel_x]
    return output_data


def cal_psnr(gt, result, white_level=16383):
    psnr = skimage.metrics.peak_signal_noise_ratio(
        gt.astype(np.float), result.astype(np.float), data_range=white_level)
    return psnr

def cal_ssim(gt, result, white_level=16383):
    ssim = skimage.metrics.structural_similarity(
        gt.astype(np.float), result.astype(np.float), multichannel=True, data_range=white_level)
    return ssim

def show_image(result_path1, result_path2, noise_path):
    f0 = rawpy.imread(result_path1)
    f1 = rawpy.imread(result_path2)
    f2 = rawpy.imread(noise_path)
    f, axarr = plt.subplots(1, 3)
    axarr[0].imshow(f0.postprocess(use_camera_wb=True))
    axarr[1].imshow(f1.postprocess(use_camera_wb=True))
    axarr[2].imshow(f2.postprocess(use_camera_wb=True))
    axarr[0].set_title('de-noise1')
    axarr[1].set_title('de-noise2')
    axarr[2].set_title('noisy')
    plt.show()

def denoise_raw(input_path, output_path, ground_path, model_path, black_level, white_level):
    """
    Example: obtain ground truth
    """
    gt = rawpy.imread(ground_path).raw_image_visible 

    """
    pre-process
    """
    raw_data_expand_c, height, width = read_image(input_path)
    raw_data_expand_c_normal = normalization(raw_data_expand_c, black_level, white_level)
    raw_data_expand_c_normal = torch.from_numpy(np.transpose(
        raw_data_expand_c_normal.reshape(-1, height//2, width//2, 4), (0, 3, 1, 2))).float()
    net = Unet()
    if model_path is not None:
        net.load_state_dict(torch.load(model_path))
    net.eval()

    """
    inference
    """
    result_data = net(raw_data_expand_c_normal)

    """
    post-process
    """
    result_data = result_data.cpu().detach().numpy().transpose(0, 2, 3, 1)
    result_data = inv_normalization(result_data, black_level, white_level)
    result_write_data = write_image(result_data, height, width)
    write_back_dng(input_path, output_path, result_write_data)

    """
    obtain psnr and ssim
    """
    psnr = skimage.metrics.peak_signal_noise_ratio(
        gt.astype(np.float), result_write_data.astype(np.float), data_range=white_level)
    ssim = skimage.metrics.structural_similarity(
        gt.astype(np.float), result_write_data.astype(np.float), multichannel=True, data_range=white_level)
    print('psnr:', psnr)
    print('ssim:', ssim)

    """
    Example: this demo_code shows your input or gt or result image
    """
    f0 = rawpy.imread(ground_path)
    f1 = rawpy.imread(input_path)
    f2 = rawpy.imread(output_path)
    f, axarr = plt.subplots(1, 3)
    axarr[0].imshow(f0.postprocess(use_camera_wb=True))
    axarr[1].imshow(f1.postprocess(use_camera_wb=True))
    axarr[2].imshow(f2.postprocess(use_camera_wb=True))
    axarr[0].set_title('gt')
    axarr[1].set_title('noisy')
    axarr[2].set_title('de-noise')


def main(args):
    model_path = args.model_path
    black_level = args.black_level
    white_level = args.white_level
    input_path = args.input_path
    output_path = args.output_path
    ground_path = args.ground_path

    denoise_raw(input_path, output_path, ground_path, model_path, black_level, white_level)


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--model_path', type=str, default="./models/th_model.pth")
    # parser.add_argument('--black_level', type=int, default=1024)
    # parser.add_argument('--white_level', type=int, default=16383)
    # parser.add_argument('--input_path', type=str, default="./data/noise/demo_noise.dng")
    # parser.add_argument('--output_path', type=str, default="./data/result/demo_torch_res.dng")
    # parser.add_argument('--ground_path', type=str, default="./data/gt/demo.dng")
    #
    # args = parser.parse_args()
    # main(args)

    # gt_path = r'D:\zdy\3_interest\2022_denoise\dataset\1_noise.dng'
    result_path1 = r'D:\zdy\3_interest\2022_denoise\dataset\results\SCUNet\denoise0.dng'
    result_path2 = r'D:\zdy\3_interest\2022_denoise\dataset\results\NAF\denoise0.dng'
    # result_path3 = r'D:\zdy\3_interest\2022_denoise\dataset\results\unet\denoise0.dng'
    # noise_path = r'D:\zdy\3_interest\2022_denoise\dataset\noisy\1_noise.dng'
    noise_path = r'D:\zdy\3_interest\2022_denoise\dataset\testset\noisy0.dng'
    show_image(result_path1, result_path2, noise_path)
    # raw_data_expand_c, height, width = read_image(gt_path)
    # print(raw_data_expand_c.shape, height, width)