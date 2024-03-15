 
import cv2
import os
import random
import torch.utils.data as data
import torch
import torchvision.transforms as transforms
from tqdm import tqdm


def prepare_Rain200H(datapath):
    input_path = os.path.join(datapath, 'rain/X2')
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

def prepare_Rain1400(datapath):
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

def prepare_Rain1200(datapath):
    # rainy and gt are connected together (left: rain, right gt)
    # imgs = []
    gts = []
    for i in range(12000):
        target_file = "%d.jpg" % (i+1)
        gts.append(os.path.join(datapath, target_file))
    print("process DID! total length: ", len(gts))
    return gts, gts

def prepare_SPA(datapath):
    print("process SPA!")
    imgs = []
    gts = []
    inputpath = os.path.join(datapath, 'input')
    gtpath = os.path.join(datapath, 'groundtruth')
    for i in range(638464):
        target_file = "%d.png" % (i + 1)
        input_file = "%d.png" % (i + 1)
        imgs.append(os.path.join(inputpath, input_file))
        gts.append(os.path.join(gtpath, target_file))
    return imgs, gts

class ContrastiveMixDataLoaderTrain(data.Dataset):
    def __init__(self, opt):
        super(ContrastiveMixDataLoaderTrain, self).__init__()
        self.opt = opt
        all_imgs, all_gts = [], []
        for data_path in opt.data_paths.split(","):
            data_path = data_path.strip()
            if data_path.find('Rain200H') != -1:
                imgs, gts = prepare_Rain200H(data_path)
            elif data_path.find('Rain200L') != -1:
                imgs, gts = prepare_Rain200L(data_path)
            elif data_path.find('Rain800') != -1:
                imgs, gts = prepare_Rain800(data_path)
            elif data_path.find('Rain1400') != -1:
                imgs, gts = prepare_Rain1400(data_path)
            elif data_path.find('Rain1200') != -1:
                imgs, gts = prepare_Rain1200(data_path)
            elif data_path.find('sparain') != -1:
                imgs, gts = prepare_SPA(data_path)
            else:
                raise (RuntimeError('Cannot find dataset!'))
            all_imgs.append(imgs)
            all_gts.append(gts)

        if len(all_imgs) == 0:
            raise(RuntimeError("Found 0 images in: " + opt.base_dir + "\n"))
        self.imgs = [i for dset in all_imgs for i in dset]
        self.gts = [i for dset in all_gts for i in dset]
        self.sizex = len(self.imgs)
        # pre-load image        
        self.imgs_numpy, self.gts_numpy = [], []
        for i in tqdm(range(self.sizex), ncols=80):
            img, gt = self.imgs[i], self.gts[i]
            img, gt = cv2.imread(img), cv2.imread(gt)
            img, gt = cv2.cvtColor(img, cv2.COLOR_BGR2RGB), cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
            self.imgs_numpy.append(img), self.gts_numpy.append(gt)

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

        self.resize = transforms.Resize(size=(opt.crop_size, opt.crop_size))

    def __len__(self):
        tot_num = self.sizex * self.opt.aug_times
        if self.opt.model_name == "IDT":
            return 1000 * self.opt.batch_size # (400 epochs: 400k iters)
        if tot_num % self.opt.batch_size == 0:
            return tot_num
        return (tot_num // self.opt.batch_size + 1) * self.opt.batch_size

    def __getitem__(self, index):
        index_ = index % self.sizex
        inp_path = self.imgs[index_]
        inp_img, tar_img = self.imgs_numpy[index_], self.gts_numpy[index_]
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

if __name__ == "__main__":
    import argparse
    import matplotlib.pyplot as plt
    opt = argparse.ArgumentParser()
    opt = opt.parse_args()
    opt.data_paths = "/home/rw/Public/datasets/derain/Rain200H/train, /home/rw/Public/datasets/derain/Rain800/train"
    opt.img_size = 128
    opt.crop_size = 128
    opt.batch_size = 16
    dset = ContrastiveMixDataLoaderTrain(opt)
    """
    for _, data in enumerate(dset):
        r, g, resize_1, resize_2, _ = data
        r = r.permute(1, 2, 0)
        g = g.permute(1, 2, 0)
        resize_1, resize_2 = resize_1.permute(1, 2, 0), resize_2.permute(1, 2, 0)
        print(r.min(), r.max(), g.min(), g.max(), resize_1.min(), resize_1.max())
        
        plt.subplot(2, 2, 1)
        plt.imshow(r.numpy())
        plt.subplot(2, 2, 2)
        plt.imshow(g.numpy())
        plt.subplot(2, 2, 3)
        plt.imshow(resize_1.numpy())
        plt.subplot(2, 2, 4)
        plt.imshow(resize_2.numpy())
        plt.show()
        """