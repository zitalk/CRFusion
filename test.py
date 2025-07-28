import torch
import torch.nn.functional as F
import sys
sys.path.append('./models')
import numpy as np
import os, argparse
import cv2

from models.MambaNet_paper import CRFusion

from data import test_dataset

parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=384, help='testing size')
parser.add_argument('--gpu_id', type=str, default='0', help='select gpu id')
opt = parser.parse_args()


#set device for test
if opt.gpu_id=='0':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    print('USE GPU 0')
elif opt.gpu_id=='1':
    os.environ["CUDA_VISIBLE_DEVICES"] = "7"
    print('USE GPU 1')

#load the model
model = CRFusion()

# model.load_state_dict(torch.load('./cpts/LightFieldNet_epoch_best.pth'), strict=False)
pth_root = './Results/CRFusion_DUTLF/best.pth'
save_path = './Results/CRFusion_DUTLF/' + os.path.basename(pth_root).split('.')[0] + '/'

model.load_state_dict(torch.load(pth_root), strict=True)
model.cuda()
model.eval()

# test_datasets = ['DUT-RGBD', 'SSD','ReDWeb','DES','LFSD','NJU2K','NLPR','SIP','STERE']


DUTLF_image_root = "/home/b311/data2/Datasets/DUTLF-V2/Test/test_images/"
DUTLF_gt_root = "/home/b311/data2/Datasets/DUTLF-V2/Test/test_masks/"
DUTLF_depth_root = "/home/b311/data2/Datasets/DUTLF-V2/Test/test_ESI/"

PKU_image_root = "/home/b311/data2/Datasets/PKU-LF/all-focus-image/test/"
PKU_gt_root = "/home/b311/data2/Datasets/PKU-LF/annotation/object/test/"
PKU_depth_root = "/home/b311/data2/Datasets/PKU-LF/ESI/test/"


# test_datasets = ['DUTLF', 'PKULF']
test_datasets = ['DUTLF']
for dataset in test_datasets:
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if dataset == 'DUTLF':
        image_root = DUTLF_image_root
        gt_root = DUTLF_gt_root
        depth_root = DUTLF_depth_root
    elif dataset == 'PKULF':
        image_root = PKU_image_root
        gt_root = PKU_gt_root
        depth_root = PKU_depth_root
    else:
        raise ValueError("Dataset not supported: {}".format(dataset))
    
    test_loader = test_dataset(image_root, gt_root, depth_root, opt.testsize)
    
    for i in range(test_loader.size):
        image, gt, depth, name, image_for_post = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()
        # print('depth shape: ', depth.shape)
        depth = depth = depth.repeat(1,3,1,1).cuda()
        # print('depth shape: ', depth.shape)
        # exit()
        res, res2, res3, res4 = model(image,depth)
        res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        print('save img to: ',save_path+name)
        cv2.imwrite(save_path + name, res*255)
    print('Test Done!')
