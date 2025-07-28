import os
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import ImageEnhance


# several data augumentation strategies
def cv_random_flip(img, label, depth):
    flip_flag = random.randint(0, 1)
    # flip_flag2= random.randint(0,1)
    # left right flip
    if flip_flag == 1:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        label = label.transpose(Image.FLIP_LEFT_RIGHT)
        depth = depth.transpose(Image.FLIP_LEFT_RIGHT)
    # top bottom flip
    # if flip_flag2==1:
    #     img = img.transpose(Image.FLIP_TOP_BOTTOM)
    #     label = label.transpose(Image.FLIP_TOP_BOTTOM)
    #     depth = depth.transpose(Image.FLIP_TOP_BOTTOM)
    return img, label, depth

def randomCrop(image, label, depth):
    border = 30
    image_width = image.size[0]
    image_height = image.size[1]
    crop_win_width = np.random.randint(image_width - border, image_width)
    crop_win_height = np.random.randint(image_height - border, image_height)
    random_region = (
        (image_width - crop_win_width) >> 1, (image_height - crop_win_height) >> 1, (image_width + crop_win_width) >> 1,
        (image_height + crop_win_height) >> 1)
    return image.crop(random_region), label.crop(random_region), depth.crop(random_region)


def randomRotation(image, label, depth):
    mode = Image.BICUBIC
    if random.random() > 0.8:
        random_angle = np.random.randint(-15, 15)
        image = image.rotate(random_angle, mode)
        label = label.rotate(random_angle, mode)
        depth = depth.rotate(random_angle, mode)
    return image, label, depth


def colorEnhance(image):
    bright_intensity = random.randint(5, 15) / 10.0
    image = ImageEnhance.Brightness(image).enhance(bright_intensity)
    contrast_intensity = random.randint(5, 15) / 10.0
    image = ImageEnhance.Contrast(image).enhance(contrast_intensity)
    color_intensity = random.randint(0, 20) / 10.0
    image = ImageEnhance.Color(image).enhance(color_intensity)
    sharp_intensity = random.randint(0, 30) / 10.0
    image = ImageEnhance.Sharpness(image).enhance(sharp_intensity)
    return image


def randomGaussian(image, mean=0.1, sigma=0.35):
    def gaussianNoisy(im, mean=mean, sigma=sigma):
        for _i in range(len(im)):
            im[_i] += random.gauss(mean, sigma)
        return im

    img = np.asarray(image)
    width, height = img.shape
    img = gaussianNoisy(img[:].flatten(), mean, sigma)
    img = img.reshape([width, height])
    return Image.fromarray(np.uint8(img))


def randomPeper(img):
    img = np.array(img)
    noiseNum = int(0.0015 * img.shape[0] * img.shape[1])
    for i in range(noiseNum):

        randX = random.randint(0, img.shape[0] - 1)

        randY = random.randint(0, img.shape[1] - 1)

        if random.randint(0, 1) == 0:

            img[randX, randY] = 0

        else:

            img[randX, randY] = 255
    return Image.fromarray(img)


# dataset for training
# The current loader is not using the normalized depth maps for training and test. If you use the normalized depth maps
# (e.g., 0 represents background and 1 represents foreground.), the performance will be further improved.
class SalObjDataset(data.Dataset):
    def __init__(self, image_root, gt_root, depth_root, trainsize):
        
        self.trainsize = trainsize
        # self.images = None
        # self.gts = None
        # self.depths = None
        
        #(1)PKULF
        # PKULF_root_path_1 = "/home/b311/data2/Datasets/PKU-LF/all-focus-image/train/"
        # PKULF_root_path_2 = "/home/b311/data2/Datasets/PKU-LF/annotation/object/train/"
        # PKULF_root_path_3 = "/home/b311/data2/Datasets/PKU-LF/ESI/train/"

        # self.PKULF_images = [PKULF_root_path_1 + f for f in os.listdir(PKULF_root_path_1) if f.endswith('.jpg')]
        # self.PKULF_gts = [PKULF_root_path_2 + f for f in os.listdir(PKULF_root_path_2) if f.endswith('.jpg') or f.endswith('.png')]
        # self.PKULF_depths = [PKULF_root_path_3 + f for f in os.listdir(PKULF_root_path_3) if f.endswith('.bmp') or f.endswith('.png') or f.endswith('.jpg')]

        # self.images = self.PKULF_images
        # self.gts = self.PKULF_gts
        # self.depths = self.PKULF_depths


        #(2)DUTLFV2
        DUTLFV2_root_path_1 = "/home/b311/data2/Datasets/DUTLF-V2/Train/train_array_No_augmentation/"
        DUTLFV2_root_path_2 = "/home/b311/data2/Datasets/DUTLF-V2/Train/train_masks/"
        DUTLFV2_root_path_3 = "/home/b311/data2/Datasets/DUTLF-V2/Train/train_ESI/"

        self.DUTLFV2_images = [DUTLFV2_root_path_1 + f for f in os.listdir(DUTLFV2_root_path_1) if f.endswith('.jpg')]
        self.DUTLFV2_gts = [DUTLFV2_root_path_2 + f for f in os.listdir(DUTLFV2_root_path_2) if f.endswith('.jpg') or f.endswith('.png')]
        self.DUTLFV2_depths = [DUTLFV2_root_path_3 + f for f in os.listdir(DUTLFV2_root_path_3) if f.endswith('.bmp') or f.endswith('.png') or f.endswith('.jpg')]
        
        self.images = self.DUTLFV2_images
        self.gts = self.DUTLFV2_gts
        self.depths = self.DUTLFV2_depths
        

        #(3)RGB-D
        # NJUNLPR_root_path_1 = "/home/b311/data2/zzx/Explicit-Visual-Prompt/data/RGBD_dataset/train/NJUNLPR/RGB/"
        # NJUNLPR_root_path_2 = "/home/b311/data2/zzx/Explicit-Visual-Prompt/data/RGBD_dataset/train/NJUNLPR/GT/"
        # NJUNLPR_root_path_3 = "/home/b311/data2/zzx/Explicit-Visual-Prompt/data/RGBD_dataset/train/NJUNLPR/depth/"

        # self.NJUNLPR_images = [NJUNLPR_root_path_1 + f for f in os.listdir(NJUNLPR_root_path_1) if f.endswith('.jpg')]
        # self.NJUNLPR_gts = [NJUNLPR_root_path_2 + f for f in os.listdir(NJUNLPR_root_path_2) if f.endswith('.jpg') or f.endswith('.png')]
        # self.NJUNLPR_depths = [NJUNLPR_root_path_3 + f for f in os.listdir(NJUNLPR_root_path_3) if f.endswith('.bmp') or f.endswith('.png') or f.endswith('.jpg')]

        #(4)RGB-T
        # self.rgbt_images = "/home/b311/data2/zzx/Explicit-Visual-Prompt/data/RGBT_dataset/train/RGB/"
        # self.rgbt_gts = "/home/b311/data2/zzx/Explicit-Visual-Prompt/data/RGBT_dataset/train/GT/"
        # self.rgbt_t = "/home/b311/data2/zzx/Explicit-Visual-Prompt/data/RGBT_dataset/train/T/"
        
        # self.rgbt_images = [self.rgbt_images + f for f in os.listdir(self.rgbt_images) if f.endswith('.jpg')]
        # self.rgbt_gts = [self.rgbt_gts + f for f in os.listdir(self.rgbt_gts) if f.endswith('.jpg') or f.endswith('.png')]
        # self.rgbt_t = [self.rgbt_t + f for f in os.listdir(self.rgbt_t) if f.endswith('.bmp') or f.endswith('.png') or f.endswith('.jpg')]

        # self.images = self.rgbt_images
        # self.gts = self.rgbt_gts
        # self.depths = self.rgbt_t

        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.depths = sorted(self.depths)
        print(len(self.images), len(self.gts), len(self.depths))
        assert len(self.images) > 0, "No images found in the specified directory."
        assert len(self.gts) > 0, "No ground truth images found in the specified directory."
        assert len(self.depths) > 0, "No depth images found in the specified directory."
        assert len(self.images) == len(self.gts) and len(self.gts) == len(self.depths), \
            "The number of images, ground truth images, and depth images must be the same."
        # filter the files

        # self.filter_files()


        # exit()

        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.depths = sorted(self.depths)
        print(len(self.images), len(self.gts), len(self.depths))
        assert len(self.images) > 0, "No images found in the specified directory."
        assert len(self.gts) > 0, "No ground truth images found in the specified directory."
        assert len(self.depths) > 0, "No depth images found in the specified directory."
        assert len(self.images) == len(self.gts) and len(self.gts) == len(self.depths), \
            "The number of images, ground truth images, and depth images must be the same."

        self.size = len(self.images)
        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])
        self.depths_transform = transforms.Compose(
            [transforms.Resize((self.trainsize, self.trainsize)), transforms.ToTensor()])


    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])
        depth = self.binary_loader(self.depths[index])
        #
        w, h = depth.size
        image = image.resize((w, h), Image.BILINEAR)
        gt = gt.resize((w, h), Image.BILINEAR)
        #


        image, gt, depth = cv_random_flip(image, gt, depth)
        image, gt, depth = randomCrop(image, gt, depth)
        image, gt, depth = randomRotation(image, gt, depth)
        image = colorEnhance(image)
        # gt=randomGaussian(gt)
        gt = randomPeper(gt)
        image = self.img_transform(image)
        gt = self.gt_transform(gt)
        depth = self.depths_transform(depth)

        return image, gt, depth

    def filter_files(self):
        assert len(self.images) == len(self.gts) and len(self.gts) == len(self.images)
        images = []
        gts = []
        depths = []
        for img_path, gt_path, depth_path in zip(self.images, self.gts, self.depths):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            depth = Image.open(depth_path)
            print(img.size, gt.size, depth.size)
            if img.size == gt.size and gt.size == depth.size:
                images.append(img_path)
                gts.append(gt_path)
                depths.append(depth_path)
        self.images = images
        self.gts = gts
        self.depths = depths

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def resize(self, img, gt, depth):
        assert img.size == gt.size and gt.size == depth.size
        w, h = img.size
        if h < self.trainsize or w < self.trainsize:
            h = max(h, self.trainsize)
            w = max(w, self.trainsize)
            return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST), \
                   depth.resize((w, h),Image.NEAREST)
        else:
            return img, gt, depth

    def __len__(self):
        return self.size





# class MEPISalObjDataset(data.Dataset):
#     def __init__(self, image_root, gt_root, depth_root, trainsize):
        
#         self.trainsize = trainsize
#         # self.images = None
#         # self.gts = None
#         # self.depths = None
        
#         #(1)PKULF
#         # PKULF_root_path_1 = "/home/b311/data2/Datasets/PKU-LF/all-focus-image/train/"
#         # PKULF_root_path_2 = "/home/b311/data2/Datasets/PKU-LF/annotation/object/train/"
#         # PKULF_root_path_3 = "/home/b311/data2/Datasets/PKU-LF/ESI/train/"

#         # self.PKULF_images = [PKULF_root_path_1 + f for f in os.listdir(PKULF_root_path_1) if f.endswith('.jpg')]
#         # self.PKULF_gts = [PKULF_root_path_2 + f for f in os.listdir(PKULF_root_path_2) if f.endswith('.jpg') or f.endswith('.png')]
#         # self.PKULF_depths = [PKULF_root_path_3 + f for f in os.listdir(PKULF_root_path_3) if f.endswith('.bmp') or f.endswith('.png') or f.endswith('.jpg')]

#         # self.images = self.PKULF_images
#         # self.gts = self.PKULF_gts
#         # self.depths = self.PKULF_depths


#         #(2)DUTLFV2
#         DUTLFV2_root_path_1 = "/home/b311/data2/zzx/Explicit-Visual-Prompt/data/DUTLF/train/RGB/"
#         DUTLFV2_root_path_2 = "/home/b311/data2/zzx/Explicit-Visual-Prompt/data/DUTLF/train/GT/"
#         DUTLFV2_root_path_3 = "/home/b311/data2/WMZ/MEPI/DUTLF-V2/Train/train_array_No_augmentation/"

#         self.DUTLFV2_images = [DUTLFV2_root_path_1 + f for f in os.listdir(DUTLFV2_root_path_1) if f.endswith('.jpg')]
#         self.DUTLFV2_gts = [DUTLFV2_root_path_2 + f for f in os.listdir(DUTLFV2_root_path_2) if f.endswith('.jpg') or f.endswith('.png')]
#         self.DUTLFV2_h_depths = [DUTLFV2_root_path_3 + f for f in os.listdir(DUTLFV2_root_path_3) if f.endswith('im_h.jpg')]
#         self.DUTLFV2_v_depths = [DUTLFV2_root_path_3 + f for f in os.listdir(DUTLFV2_root_path_3) if f.endswith('im_v.jpg')]
        
#         self.images = self.DUTLFV2_images
#         self.gts = self.DUTLFV2_gts
#         self.depths_h = self.DUTLFV2_h_depths
#         self.depths_v = self.DUTLFV2_v_depths

#         self.images = sorted(self.images)
#         self.gts = sorted(self.gts)
#         self.depths_h = sorted(self.depths_h)
#         self.depths_v = sorted(self.depths_v)
#         print(len(self.images), len(self.gts), len(self.depths_h)), print(len(self.depths_v))
#         assert len(self.images) > 0, "No images found in the specified directory."
#         assert len(self.gts) > 0, "No ground truth images found in the specified directory."
#         assert len(self.depths_h) > 0, "No depth images found in the specified directory."
#         assert len(self.depths_v) > 0, "No depth images found in the specified directory."
#         assert len(self.images) == len(self.gts) and len(self.gts) == len(self.depths_h) == len(self.depths_v), \
#             "The number of images, ground truth images, and depth images must be the same."
#         # filter the files

#         self.size = len(self.images)
#         self.img_transform = transforms.Compose([
#             transforms.Resize((self.trainsize, self.trainsize)),
#             transforms.ToTensor(),
#             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
#         self.gt_transform = transforms.Compose([
#             transforms.Resize((self.trainsize, self.trainsize)),
#             transforms.ToTensor()])
#         self.depths_transform = transforms.Compose(
#             [transforms.Resize((self.trainsize, self.trainsize)), transforms.ToTensor()])


#     def __getitem__(self, index):
#         image = self.rgb_loader(self.images[index])
#         gt = self.binary_loader(self.gts[index])
#         depth_h = self.binary_loader(self.depths_h[index])
#         depth_v = self.binary_loader(self.depths_v[index])
#         print("图像输入大小：",image.size, gt.size, depth_h.size, depth_v.size)

#         exit()
#         image, gt, depth = cv_random_flip(image, gt, depth)
#         image, gt, depth = randomCrop(image, gt, depth)
#         image, gt, depth = randomRotation(image, gt, depth)
#         image = colorEnhance(image)
#         # gt=randomGaussian(gt)
#         gt = randomPeper(gt)
#         image = self.img_transform(image)
#         gt = self.gt_transform(gt)
#         depth = self.depths_transform(depth)

#         return image, gt, depth


#     def rgb_loader(self, path):
#         with open(path, 'rb') as f:
#             img = Image.open(f)
#             return img.convert('RGB')

#     def binary_loader(self, path):
#         with open(path, 'rb') as f:
#             img = Image.open(f)
#             return img.convert('L')

#     def resize(self, img, gt, depth):
#         assert img.size == gt.size and gt.size == depth.size
#         w, h = img.size
#         if h < self.trainsize or w < self.trainsize:
#             h = max(h, self.trainsize)
#             w = max(w, self.trainsize)
#             return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST), \
#                    depth.resize((w, h),Image.NEAREST)
#         else:
#             return img, gt, depth

#     def __len__(self):
#         return self.size






# dataloader for training
def get_loader(image_root, gt_root, depth_root, batchsize, trainsize, shuffle=True, num_workers=8, pin_memory=True, MEPI=False):
    if MEPI:
        dataset = MEPISalObjDataset(image_root, gt_root, depth_root, trainsize)
    else:
        dataset = SalObjDataset(image_root, gt_root, depth_root, trainsize)
    # print(image_root)
    # print(gt_root)
    # print(depth_root)
    # print("--------------------------------------------------------------------------------------------")
    # print(dataset)
    # print("--------------------------------------------------------------------------------------------")
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader


# test dataset and loader
class test_dataset:
    def __init__(self, image_root, gt_root, depth_root, testsize):
        self.testsize = testsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg')
                    or f.endswith('.png')]
        self.depths = [depth_root + f for f in os.listdir(depth_root) if f.endswith('.bmp')
                       or f.endswith('.png')or f.endswith('.jpg')]

        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.depths = sorted(self.depths)
        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.ToTensor()
        # self.gt_transform = transforms.Compose([
        #     transforms.Resize((self.trainsize, self.trainsize)),
        #     transforms.ToTensor()])
        self.depths_transform = transforms.Compose(
            [transforms.Resize((self.testsize, self.testsize)), transforms.ToTensor()])

        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        gt = self.binary_loader(self.gts[self.index])
        depth = self.binary_loader(self.depths[self.index])

        #
        w, h = depth.size
        image = image.resize((w, h), Image.BILINEAR)
        gt = gt.resize((w, h), Image.BILINEAR)
        #
        # gt.save("./test_gt.png")
        # depth.save("./test_depth.png")
        image = self.transform(image).unsqueeze(0)
        depth = self.depths_transform(depth).unsqueeze(0)
        name = self.gts[self.index].split('/')[-1]
        image_for_post = self.rgb_loader(self.images[self.index])
        image_for_post = image_for_post.resize(gt.size)
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.jpg'
        self.index += 1
        self.index = self.index % self.size
        return image, gt, depth, name, np.array(image_for_post)

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def __len__(self):
        return self.size

