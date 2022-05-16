import argparse
import os
import numpy as np
import rawpy, cv2
# from data_utils import read_image
import glob
from tqdm import tqdm
def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', type=str, default='/codec/zhangdy/datasets/dataset/')
    parser.add_argument('--out_path', type=str, default='/codec/zhangdy/datasets/dataset/gl_npy')
    parser.add_argument('--out_path_val', type=str, default='../dataset/dataset_crop_val')
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
    resize_img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_NEAREST)
    print(os.path.join(save_dir, img_name.replace('.dng', '.npy')))
    #np.save(os.path.join(save_dir, img_name.replace('.dng', '.npy')), resize_img)

    return 'Processing {:s} ...'.format(img_name)

def make_data(opt):
    #paths_clean = sorted(glob.glob(os.path.join(opt.dataroot, 'ground_truth', '*.dng')))#[:2]
    #for i, file in enumerate(paths_clean):
        #save_dir = os.path.join(opt.out_path, 'ground_truth_npy', str(i))
        #print(save_dir)
        #os.makedirs(save_dir, exist_ok=True)
        #data = read_image(file)[0]
        #worker(data, save_dir, file, fg='train')

    paths_noise = sorted(glob.glob(os.path.join(opt.dataroot, 'noisy', '*.dng')))#[:2]
    for i, file in enumerate(paths_noise):
        save_dir = os.path.join(opt.out_path, 'noisy_npy', str(i))
        print(save_dir)
        os.makedirs(save_dir, exist_ok=True)
        data = read_image(file)[0]
        worker(data, save_dir, file, fg='train')

if __name__ == '__main__':
    opt = parser_args()
    make_data(opt)

