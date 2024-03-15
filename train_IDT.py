"""
Train DGUNet with mixed datasets following Low-Rank Parameter Adaptation strategy
"""
from tensorboardX import SummaryWriter
import argparse
import numpy as np
import os
import torch
from torch.backends import cudnn
import random
import torch.nn.functional as F
from evaluation import psnr as compare_psnr
import shutil
from models.select_model import define_model
from models.contrastive import MoCo
from common_datasets.mix_dataset_global import getcontrastivemixloader
from torch.optim.lr_scheduler import MultiStepLR
from SSIM import SSIM


class Experiments:
    def __init__(self, opt):
        ## Initialize dataloader
        self.opt = opt
        self.dataloader_train = getcontrastivemixloader(opt)
        self.epochs = opt.epochs
        self.device = torch.device('cuda') if opt.gpu_id else torch.device('cpu')
        print('# of training samples: %d \n' % int(len(self.dataloader_train.dataset)))
        # Build Model
        self.model = define_model(opt=opt)
        self.feat_extractor = MoCo(in_channels=3, out_channels=32, dim=128, temperature=opt.temperature)
        self.model.to(self.device)
        self.feat_extractor.to(self.device)
        # criterion
        self.criterion = SSIM()
        self.criterion_contra = torch.nn.CrossEntropyLoss()
        base_parameters, tran_parameters = [], []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if 'g2pa' in name:
                    tran_parameters.append(param)
                else:
                    base_parameters.append(param)
        tran_parameters.extend(self.feat_extractor.parameters()) # append parameters from feature extractor
        # Initialize optimizers and schedulers
        self.base_optimizer = torch.optim.Adam(params=base_parameters, lr=opt.lr, betas=(0.9, 0.999))
        self.tran_optimizer = torch.optim.Adam(params=tran_parameters, lr=opt.lr, betas=(0.9, 0.999))
        self.base_scheduler = MultiStepLR(self.base_optimizer, milestones=opt.milestone, gamma=0.1)
        self.tran_scheduler = MultiStepLR(self.tran_optimizer, milestones=opt.milestone, gamma=0.1) 
        # Create log folder
        os.makedirs(opt.save_path, exist_ok=True)
        self.writter = SummaryWriter(logdir=opt.save_path)
        self.writter.add_text(tag="opt", text_string=str(opt))
        self.init_epoch = 0
        # save files
        shutil.copy("common_datasets/mix_dataset_global.py", os.path.join(self.opt.save_path, "dataset.py"))
        shutil.copy("models/contrastive.py", os.path.join(self.opt.save_path, "networks.py"))
        shutil.copy(__file__, os.path.join(self.opt.save_path, "train.py"))
        # Load latest checkpoint if exists
        if os.path.exists(os.path.join(opt.save_path, 'latest.tar')):
            self.init_epoch = self.load_checkpoint(os.path.join(self.opt.save_path, 'latest.tar'))
    
    def load_checkpoint(self, ckp_path):
        """
        Load checkpoint from ckp_path
        :param: obtain_epoch: obtain current_epoch in last training process if interrupted
        """
        ckp = torch.load(ckp_path)
        self.model.load_state_dict(ckp['base_state_dict'])
        self.feat_extractor.load_state_dict(ckp['tran_state_dict'])
        self.base_optimizer.load_state_dict(ckp['base_optim'])
        self.tran_optimizer.load_state_dict(ckp['tran_optim'])
        return int(ckp['epoch']) 

    def train(self):
        # Start training
        step = (self.init_epoch)*len(self.dataloader_train)
        for epoch in range(self.init_epoch, self.epochs):
            for param_group in self.base_optimizer.param_groups:
                self.writter.add_scalar(tag="base_lr", scalar_value=param_group["lr"], global_step=1+epoch)
            for param_group in self.tran_optimizer.param_groups:
                self.writter.add_scalar(tag="tran_lr", scalar_value=param_group["lr"], global_step=1+epoch)
            self.feat_extractor.train()
            self.model.train()

            for iter, (input_train, target_train, resize_inps, resize_tars) in enumerate(self.dataloader_train):
                input_train = input_train.to(self.device, non_blocking=True).float() / 255.0
                target_train = target_train.to(self.device, non_blocking=True) / 255.0
                # postive and negative anchors
                im_q = resize_inps.to(self.device, non_blocking=True).float() / 255.0
                im_k = self.dataloader_train.dataset.k_transform(im_q)  # batch transform
                im_negs = resize_tars.to(self.device, non_blocking=True).float() / 255.0
                # print(im_negs.shape, im_q.shape, im_k.shape)
                im_negs = self.dataloader_train.dataset.neg_transform(im_negs.unsqueeze(1).repeat(1, opt.n_neg, 1, 1, 1).reshape(opt.batch_size*opt.n_neg, 3, 
                                                                                              opt.crop_size, opt.crop_size)) # batch transform
                im_negs = im_negs.reshape(opt.batch_size, opt.n_neg, 3, opt.crop_size, opt.crop_size)
                
                im_rain = (input_train - target_train)
                diff_rain = (im_rain.unsqueeze(1) - im_rain.unsqueeze(0)).abs().sum(dim=[2, 3, 4])
                max_diff_index = torch.argmax(diff_rain, dim=-1)
                max_diff_rain = im_rain[max_diff_index] # [B, 3, H, W]
                max_diff_rain = max_diff_rain.unsqueeze(1) # [B, 1, 3, H, W]
                im_negs = torch.cat([(max_diff_rain + target_train.unsqueeze(1)).clamp_(0.0, 1.0), im_negs], dim=1)
                self.base_optimizer.zero_grad()
                self.tran_optimizer.zero_grad()
                if step <= 4000000:
                    outs = self.model(input_train, mode="normal")
                    base_loss = - self.criterion(outs, target_train)
                    contra_loss = 0.0*base_loss
                    (base_loss).backward()
                else:
                    logits, labels, z_q = self.feat_extractor(im_q, im_k, im_negs)
                    contra_loss = self.criterion_contra(logits, labels)
                    outs = self.model(input_train, context=z_q, mode="tran")
                    base_loss = - self.criterion(outs, target_train)
                    contra_loss *= self.opt.contra_loss_weight
                    (base_loss + contra_loss).backward()
                self.base_optimizer.step()
                self.tran_optimizer.step()

                if (1+iter) % 40 == 0:
                    out_train = torch.clamp(outs.detach().cpu(), 0.0, 1.0)
                    psnr_train = compare_psnr(out_train, target_train.cpu(), data_range=1.0)
                    self.writter.add_scalar("base_loss", base_loss.item(), step)
                    self.writter.add_scalar("contra_loss", contra_loss.item(), step)
                    self.writter.add_scalar("psnr", psnr_train, step)
                    if step <= 4000000:
                        msg = 'epoch {:03d}/{:03d}, [{:03d}/{:03d}] | base_loss: {:6f} | contra_loss: {:6f} | psnr: {:4f}'.format(epoch, self.epochs, iter, len(self.dataloader_train), 
                                    base_loss.item(), 0.0, psnr_train)
                    else:
                        msg = 'epoch {:03d}/{:03d}, [{:03d}/{:03d}] | base_loss: {:6f} | contra_loss: {:6f} | psnr: {:4f}'.format(epoch, self.epochs, iter, len(self.dataloader_train), 
                                    base_loss.item(), contra_loss.item(), psnr_train)

                    print(msg)
                step += 1
                if step == 40000:
                    torch.save({
                        'epoch': epoch,
                        'base_state_dict': self.model.state_dict(),
                        'tran_state_dict': self.feat_extractor.state_dict(),
                        'base_optim': self.base_optimizer.state_dict(),
                        'tran_optim': self.tran_optimizer.state_dict(),
                    }, os.path.join(self.opt.save_path, 'latest_40k.tar'))
            # learning rate scheduler
            self.base_scheduler.step(epoch)
            self.tran_scheduler.step(epoch)
            # save_model
            torch.save(self.model.state_dict(), os.path.join(self.opt.save_path, 'net_epoch_{}.pth'.format(epoch)))
            torch.save(self.feat_extractor.state_dict(), os.path.join(self.opt.save_path, 'feat_epoch_{}.pth'.format(epoch)))
            torch.save({
                'epoch': epoch,
                'base_state_dict': self.model.state_dict(),
                'tran_state_dict': self.feat_extractor.state_dict(),
                'base_optim': self.base_optimizer.state_dict(),
                'tran_optim': self.tran_optimizer.state_dict(),
            }, os.path.join(self.opt.save_path, 'latest.tar'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='IDT_train')
    parser.add_argument("--batch_size", type=int, default=8, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=400, help="Number of training epochs")
    parser.add_argument("--milestone", type=int, default=[100, 250, 350], help="When to decay learning rate")
    parser.add_argument("--lr", type=float, default=1e-4, help="initial learning rate")
    parser.add_argument("--save_path", type=str, default="logs/IDT-H8L1214-baseline", help='path to save models and log files')
    parser.add_argument("--save_freq", type=int, default=1, help='save intermediate model')
    parser.add_argument("--use_GPU", action="store_true", help='use GPU or not')
    parser.add_argument("--gpu_id", type=str, default="0", help='GPU id')
    parser.add_argument("--data_paths", type=str, default="/home1/zhangsy/rh/data/derain/Rain200H/train/, \
                       /home1/zhangsy/rh/data/derain/Rain200L/train/, /home1/zhangsy/rh/data/derain/Rain800/train/,\
                        /home1/zhangsy/rh/data/derain/Rain1200_new/train, /home1/zhangsy/rh/data/derain/Rain14000/train")
    parser.add_argument("--model_name", type=str, default="IDT", help="training model name")
    parser.add_argument("--crop_size", type=int, default=128)
    parser.add_argument("--aug_times", type=int, default=1, help="augmentation times")
    parser.add_argument("--num_workers", type=int, default=16, help="number of workers")
    parser.add_argument("--seed", type=int, default=0, help='random seed')
    parser.add_argument("--dim_in", type=int, default=128, help='dimension of code z')
    parser.add_argument("--contra_loss_weight", type=float, default=0.2, help="contra_loss_weight")
    parser.add_argument("--temperature", type=float, default=1.0, help="temperature")
    parser.add_argument("--n_neg", type=int, default=4, help="number of negative examples")
    opt = parser.parse_args()

    parser = argparse.ArgumentParser(description="IDT_train")

    train_type = __file__.split("train")[-1].split(".")[0]
    dim_type = "_{}d".format(opt.dim_in)
    contra_type = "_{}contra".format(opt.contra_loss_weight)
    opt.save_path = opt.save_path + train_type + dim_type + contra_type

    os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.gpu_id)    
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

    exp = Experiments(opt=opt)
    exp.train()
