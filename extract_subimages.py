import argparse
import os
import numpy as np
import rawpy
# from data_utils import read_image
import glob

def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', type=str, default='/codec/zhangdy/datasets/dataset')
    parser.add_argument('--out_path', type=str, default='/codec/zhangdy/datasets/dataset/crop')
    parser.add_argument('--out_path_val', type=str, default='/codec/zhangdy/datasets/dataset/crop_val')
    return parser.parse_args()
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

def worker(img, save_dir, img_name, fg):
    crop_sz = 640
    if fg == 'train':
        step = 64 # 960
    if fg == 'val':
        step = 960 # 960
    thres_sz = 48
    n_channels = 4
    h, w, c = img.shape
    assert c == n_channels, 'Error channel c:{}!=4'.format(c)
    img_name = os.path.basename(img_name)
    # print(img_name)
    h_space = np.arange(0, h - crop_sz + 1, step)
    if h - (h_space[-1] + crop_sz) > thres_sz:
        h_space = np.append(h_space, h - crop_sz)
    w_space = np.arange(0, w - crop_sz + 1, step)
    if w - (w_space[-1] + crop_sz) > thres_sz:
        w_space = np.append(w_space, w - crop_sz)

    index = 0
    for x in h_space:
        for y in w_space:
            index += 1
            crop_img = img[x:x + crop_sz, y:y + crop_sz, :]
            crop_img = np.ascontiguousarray(crop_img)
            # print(os.path.join(save_dir, img_name.replace('.dng', '_s{:03d}.npy'.format(index))))
            np.save(os.path.join(save_dir, img_name.replace('.dng', '_s{:03d}.npy'.format(index))), crop_img)

    return 'Processing {:s} ...'.format(img_name)

def get_train(opt):
    paths_clean = sorted(glob.glob(os.path.join(opt.dataroot, 'ground_truth', '*.dng')))[2:]
    for i, file in enumerate(paths_clean):
        save_dir = os.path.join(opt.out_path, 'ground_truth_crop', str(i))
        print(save_dir)
        os.makedirs(save_dir, exist_ok=True)
        data = read_image(file)[0]
        worker(data, save_dir, file,fg='train')

    paths_noise = sorted(glob.glob(os.path.join(opt.dataroot, 'noisy', '*.dng')))[2:]
    for i, file in enumerate(paths_noise):
        save_dir = os.path.join(opt.out_path, 'noisy_crop', str(i))
        print(save_dir)
        os.makedirs(save_dir, exist_ok=True)
        data = read_image(file)[0]
        worker(data, save_dir, file, fg='train')

def get_val(opt):
    paths_clean = sorted(glob.glob(os.path.join(opt.dataroot, 'ground_truth', '*.dng')))[:2]
    for i, file in enumerate(paths_clean):
        save_dir = os.path.join(opt.out_path_val, 'ground_truth_crop', str(i))
        print(save_dir)
        os.makedirs(save_dir, exist_ok=True)
        data = read_image(file)[0]
        worker(data, save_dir, file, fg='val')

    paths_noise = sorted(glob.glob(os.path.join(opt.dataroot, 'noisy', '*.dng')))[:2]
    for i, file in enumerate(paths_noise):
        save_dir = os.path.join(opt.out_path_val, 'noisy_crop', str(i))
        print(save_dir)
        os.makedirs(save_dir, exist_ok=True)
        data = read_image(file)[0]
        worker(data, save_dir, file, fg='val')

if __name__ == '__main__':
    opt = parser_args()
    # main(opt)
    get_val(opt)
    get_train(opt)