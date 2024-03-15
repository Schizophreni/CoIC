 
import cv2
import os
import random
import torch.utils.data as data
import torch
import torchvision.transforms as transforms
from tqdm import tqdm
import numpy as np
import copy


def prepare_Rain200H(datapath):
    input_path = os.path.join(datapath, 'rain')
    target_path = os.path.join(datapath, 'norain')
    imgs = []
    gts = []
    for i in range(1800):
        target_file = "norain-%d.png" % (i + 1)
        input_file = "norain-%dx2.png" %(i + 1)
        imgs.append(os.path.join(input_path, input_file))
        gts.append(os.path.join(target_path, target_file))
    print("process Rain200H! total length: ", len(imgs))
    return imgs, gts

def prepare_Rain200L(datapath):
    input_path = os.path.join(datapath, 'rain')
    target_path = os.path.join(datapath, 'norain')
    imgs = []
    gts = []
    for i in range(1800):
        target_file = "norain-%d.png" % (i + 1)
        input_file = "norain-%dx2.png" %(i + 1)
        imgs.append(os.path.join(input_path, input_file))
        gts.append(os.path.join(target_path, target_file))
    print("process Rain200L! total length: ", len(imgs))
    return imgs, gts

def prepare_Rain800(datapath):
    input_path = os.path.join(datapath, 'rain')
    target_path = os.path.join(datapath, 'norain')
    imgs = []
    gts = []
    for i in range(700):
        target_file = "norain-%03d.png" % (i + 1)
        input_file = "rain-%03d.png" %(i + 1)
        imgs.append(os.path.join(input_path, input_file))
        gts.append(os.path.join(target_path, target_file))
    print("process Rain800! total length: ", len(imgs))
    return imgs, gts 

def prepare_DDN(datapath):
    input_path = os.path.join(datapath, 'rain')
    target_path = os.path.join(datapath, 'norain')
    imgs = []
    gts = []
    for i in range(900):
        target_file = "%d.jpg" % (i + 1)
        for j in range(14):
            input_file = "%d_%d.jpg" % (i + 1, j + 1)
            imgs.append(os.path.join(input_path, input_file))
            gts.append(os.path.join(target_path, target_file))
    print("process DDN! total length: ", len(imgs))
    return imgs, gts

def prepare_DID(datapath):
    # rainy and gt are connected together (left: rain, right gt)
    # imgs = []
    gts = []
    for i in range(12000):
        target_file = "%d.jpg" % (i+1)
        gts.append(os.path.join(datapath, target_file))
    print("process DID! total length: ", len(gts))
    return gts, gts

def prepare_spa(datapath):
    imgs, gts = [], []
    inputpath = os.path.join(datapath, "real_world")
    gtpath = os.path.join(datapath, "real_world_gt")
    num_scenes = len(os.listdir(inputpath))  # number of scenes
    for scene_idx in range(1, num_scenes+1):
        rain_path = os.path.join(inputpath, "{:03d}".format(scene_idx))
        gt_path = os.path.join(gtpath, "{:03d}".format(scene_idx))
        rscs = os.listdir(rain_path)  # rainy scenes
        for rsc in rscs:
            rsc_idx = rsc.split("-")[-1]
            rain_src_path = os.path.join(rain_path, "{:03d}-{}".format(scene_idx, rsc_idx))
            rain_imgs = os.listdir(rain_src_path)
            for rain_img in rain_imgs:
                x, y = rain_img.split(".")[0].split("_")[-2:]  # scene x, y coordinates
                imgs.append(os.path.join(rain_src_path, rain_img))
                gt_img = os.path.join(gt_path, "{:03d}_{}_{}.png".format(scene_idx, x, y))
                gts.append(gt_img)
    print("Total imgs: ", len(imgs), len(gts))
    return imgs, gts

class ContrastiveMixDataLoaderTrain(data.Dataset):
    def __init__(self, opt):
        super(ContrastiveMixDataLoaderTrain, self).__init__()
        self.opt = opt
        self.syn_imgs, self.syn_gts = [], []
        self.real_imgs, self.real_gts = [], []
        for data_path in opt.data_paths.split(","):
            data_path = data_path.strip()
            if data_path.find('Rain200H') != -1:
                imgs, gts = prepare_Rain200H(data_path)
            elif data_path.find('Rain200L') != -1:
                imgs, gts = prepare_Rain200L(data_path)
            elif data_path.find('Rain800') != -1:
                imgs, gts = prepare_Rain800(data_path)
            elif data_path.find('DDN') != -1:
                imgs, gts = prepare_DDN(data_path)
            elif data_path.find('DID') != -1:
                imgs, gts = prepare_DID(data_path)
            elif data_path.find('spa') != -1:
                imgs, gts = prepare_spa(data_path)
                self.real_imgs.extend(imgs)
                self.real_gts.extend(gts)
                continue
            else:
                raise (RuntimeError('Cannot find dataset!'))
            self.syn_imgs.extend(imgs)
            self.syn_gts.extend(gts)

        if len(self.syn_gts) == 0 and len(self.real_gts):
            raise(RuntimeError("Found 0 images in: " + opt.base_dir + "\n"))
        self.imgs = self.syn_imgs
        self.gts = self.syn_gts
        # add real_world images
        if len(self.real_gts) > 0:
            print("Add real world images")
            self.add_real_world()
        self.crop_size = opt.crop_size
        # pre-load image        
        self.neg_transform = transforms.Compose([
            transforms.GaussianBlur(kernel_size=11, sigma=(0.3, 1.5)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
        ])

        self.k_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
        ])
        self.sizex = len(self.imgs)

        self.resize = transforms.Resize(size=(opt.crop_size, opt.crop_size))

    def __len__(self):
        tot_num = len(self.imgs)
        if tot_num % self.opt.batch_size == 0:
            return tot_num
        return (tot_num // self.opt.batch_size + 1) * self.opt.batch_size
    
    def add_real_world(self):
        if len(self.real_imgs) == 0:
            return
        print(len(self.real_imgs))
        del self.imgs, self.gts
        self.imgs, self.gts = copy.deepcopy(self.syn_imgs), copy.deepcopy(self.syn_gts)
        real_worlds = np.random.choice(range(len(self.real_gts)), len(self.syn_imgs), replace=False)
        for idx in real_worlds:
            self.imgs.append(self.real_imgs[idx])
            self.gts.append(self.real_gts[idx])

    def __getitem__(self, index):
        index_ = index % self.sizex
        inp_path = self.imgs[index_]
        gt_path = self.gts[index_]
        # inp_img, tar_img = self.imgs_numpy[index_], self.gts_numpy[index_]
        inp_img, tar_img = cv2.imread(inp_path), cv2.imread(gt_path)
        inp_img, tar_img = cv2.cvtColor(inp_img, cv2.COLOR_BGR2RGB), cv2.cvtColor(tar_img, cv2.COLOR_BGR2RGB)
        inp_img, tar_img = torch.from_numpy(inp_img).permute(2, 0, 1).contiguous(), torch.from_numpy(tar_img).permute(2, 0, 1).contiguous()
        if "Rain1200" in inp_path:
            inp_img, tar_img = torch.chunk(inp_img, chunks=2, dim=-1)
        resize_inp, resize_tar = self.resize(inp_img), self.resize(tar_img)
        hh, ww = tar_img.shape[1], tar_img.shape[2]
        rr = random.randint(0, hh - self.crop_size)
        cc = random.randint(0, ww - self.crop_size)
        # Crop patch
        inp_img = inp_img[:, rr:rr + self.crop_size, cc:cc + self.crop_size]
        tar_img = tar_img[:, rr:rr + self.crop_size, cc:cc + self.crop_size]
        # Crop patch Data Augmentations
        aug = random.randint(0, 3)
        if aug == 1:
            inp_img = inp_img.flip(1)
            tar_img = tar_img.flip(1)
        elif aug == 2:
            inp_img = inp_img.flip(2)
            tar_img = tar_img.flip(2)
        return inp_img, tar_img, resize_inp, resize_tar

def getcontrastivemixloader(opt):
    dataset = ContrastiveMixDataLoaderTrain(opt)
    print("Dataset Size:%d" %(len(dataset)))
    trainloader = data.DataLoader(dataset,
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=opt.num_workers,
            pin_memory=True, drop_last=False)
    return trainloader
