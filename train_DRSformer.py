"""
Train DRSformer with mixed datasets following CoIC
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
from models.select_model import define_model
from models.contrastive import MoCo
from mix_dataset import getcontrastivemixloader
from torch.optim.lr_scheduler import _LRScheduler
import math
import time


## ------------------------------------------------------------|
##   First define learning rate scheduler used in DRSformer ---|
## ----------------------------------------------------------- |

# copied from github repository: https://github.com/cschenxiang/DRSformer/tree/main/basicsr
def get_position_from_periods(iteration, cumulative_period):
    """Get the position from a period list.
    It will return the index of the right-closest number in the period list.
    For example, the cumulative_period = [100, 200, 300, 400],
    if iteration == 50, return 0;
    if iteration == 210, return 2;
    if iteration == 300, return 2.
    """
    for i, period in enumerate(cumulative_period):
        if iteration <= period:
            return i

class CosineAnnealingRestartCyclicLR(_LRScheduler):
    """ Cosine annealing with restarts learning rate scheme.
    An example of config:
    periods = [10, 10, 10, 10]
    restart_weights = [1, 0.5, 0.5, 0.5]
    eta_min=1e-7
    It has four cycles, each has 10 iterations. At 10th, 20th, 30th, the
    scheduler will restart with the weights in restart_weights.
    """

    def __init__(self, optimizer, periods, restart_weights=(1, ), eta_mins=(0, ), last_epoch=-1):
        self.periods = periods
        self.restart_weights = restart_weights
        self.eta_mins = eta_mins
        assert (len(self.periods) == len(self.restart_weights)), \
                'periods and restart_weights should have the same length.'
        self.cumulative_period = [
            sum(self.periods[0:i + 1]) for i in range(0, len(self.periods))
        ]
        super(CosineAnnealingRestartCyclicLR, self).__init__(optimizer, last_epoch)
        
    def get_lr(self):
        idx = get_position_from_periods(self.last_epoch, self.cumulative_period)
        current_weight = self.restart_weights[idx]
        nearest_restart = 0 if idx == 0 else self.cumulative_period[idx - 1]
        current_period = self.periods[idx]
        eta_min = self.eta_mins[idx]

        return [
            eta_min + current_weight * 0.5 * (base_lr - eta_min) *
            (1 + math.cos(math.pi * (
                (self.last_epoch - nearest_restart) / current_period)))
            for base_lr in self.base_lrs
        ]


class Experiments:
    def __init__(self, opt):
        ## Initialize dataloader
        self.opt = opt
        self.dataloader_train = getcontrastivemixloader(opt)
        self.epochs = (opt.tot_iters // len(self.dataloader_train)) + 1
        opt.epochs = self.epochs
        self.device = torch.device('cuda') if opt.gpu_id else torch.device('cpu')
        print('# of training samples: %d \n' % int(len(self.dataloader_train.dataset)))
        # Build Model
        self.model = define_model(opt=opt)
        self.feat_extractor = MoCo(in_channels=3, out_channels=32, dim=128, temperature=opt.temperature)
        self.model.to(self.device)
        self.feat_extractor.to(self.device)
        # criterion
        self.criterion_pixel = torch.nn.L1Loss(reduction="mean")
        self.criterion_contra = torch.nn.CrossEntropyLoss()
        base_parameters, tran_parameters = [], []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if 'g2pa' in name or "context_emb" in name:
                    tran_parameters.append(param)
                else:
                    base_parameters.append(param)
        tran_parameters.extend(self.feat_extractor.parameters()) # append parameters from feature extractor
        # Initialize optimizers and schedulers
        self.base_optimizer = torch.optim.AdamW(params=base_parameters, lr=opt.lr, betas=(0.9, 0.999), weight_decay=1e-4)
        self.tran_optimizer = torch.optim.AdamW(params=tran_parameters, lr=opt.lr, betas=(0.9, 0.999))
        self.base_scheduler = CosineAnnealingRestartCyclicLR(self.base_optimizer, periods=[92000, 208000,200000], restart_weights=[1, 1, 1], 
                                                             eta_mins=[0.0003,0.0001,0.000001])
        # self.tran_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.tran_optimizer, T_max=300000, eta_min=1e-7) 
        self.tran_scheduler = CosineAnnealingRestartCyclicLR(self.tran_optimizer, periods=[92000, 208000,200000], restart_weights=[1, 1, 1],
                                                             eta_mins=[0.0003,0.0001,0.000001])
        # Create log folder
        os.makedirs(opt.save_path, exist_ok=True)
        self.writter = SummaryWriter(logdir=opt.save_path)
        self.writter.add_text(tag="opt", text_string=str(opt))
        self.init_epoch = 1
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
        print(ckp["epoch"])
        
        return int(ckp["epoch"]) + 1

    def train(self):
        # Start training
        rain_archives = []  # store recent rain for models with small batch size 
        n_archives = 64 // self.opt.batch_size
        n_recored = 0
        step = (self.init_epoch-1)*len(self.dataloader_train)
        for epoch in range(self.init_epoch, self.epochs + 1):
            for param_group in self.base_optimizer.param_groups:
                self.writter.add_scalar(tag="base_lr", scalar_value=param_group["lr"], global_step=1+epoch)
            for param_group in self.tran_optimizer.param_groups:
                self.writter.add_scalar(tag="tran_lr", scalar_value=param_group["lr"], global_step=1+epoch)
            self.feat_extractor.train()
            self.model.train()
            tic = time.time()
            self.dataloader_train.dataset.add_real_world()  # resampling 
            for iter, (input_train, target_train, resize_inps, resize_tars) in enumerate(self.dataloader_train):
                step += 1
                if step > opt.tot_iters:
                    epoch = self.epochs + 1
                    torch.save(self.model.state_dict(), os.path.join(self.opt.save_path, 'net_latest.pth'))
                    torch.save(self.feat_extractor.state_dict(), os.path.join(self.opt.save_path, 'feat_latest.pth'))
                    break
                # prepare data
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

                # construct rain data base
                if len(rain_archives) == n_archives:
                    rain_archives = rain_archives[1:]  # pop oldest one
                rain_archives.append(input_train - target_train)
                im_rain = torch.cat(rain_archives, dim=0)
                current_rain = input_train - target_train
                diff_rain = (current_rain.unsqueeze(1) - im_rain.unsqueeze(0)).abs().sum(dim=[2, 3, 4])
                max_diff_index = torch.argmax(diff_rain, dim=-1)
                max_diff_rain = im_rain[max_diff_index] # [B, 3, H, W]
                max_diff_rain = max_diff_rain.unsqueeze(1) # [B, 1, 3, H, W]
                im_negs = torch.cat([(max_diff_rain + target_train.unsqueeze(1)).clamp_(0.0, 1.0), im_negs], dim=1)
                contra_loss = 0.0

                self.base_scheduler.step(step)
                self.tran_scheduler.step(step)

                self.base_optimizer.zero_grad()
                self.tran_optimizer.zero_grad()
                if step <= self.opt.stage1_iters:
                    outs = self.model(input_train, None, mode="normal")
                    base_loss = self.criterion_pixel(outs, target_train)
                    contra_loss = base_loss * 0.0
                    (base_loss).backward()
                else:
                    logits, labels, z_q = self.feat_extractor(im_q, im_k, im_negs)
                    contra_loss = self.criterion_contra(logits, labels)
                    outs = self.model(input_train, z_q, mode="tran")
                    base_loss = self.criterion_pixel(outs, target_train)
                    contra_loss *= self.opt.contra_loss_weight
                    (base_loss + contra_loss).backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.01)  # clip grad following DRSformer
                # torch.nn.utils.clip_grad_norm_(self.feat_extractor.parameters(), 0.01)  # clip grad following DRSformer
                self.base_optimizer.step()
                self.tran_optimizer.step()
                if step % 40 == 0:
                    toc = time.time()
                    out_train = torch.clamp(outs.detach().cpu(), 0.0, 1.0)
                    psnr_train = compare_psnr(out_train, target_train.cpu(), data_range=1.0)
                    self.writter.add_scalar("base_loss", base_loss.item(), step)
                    self.writter.add_scalar("contra_loss", contra_loss.item(), step)
                    msg = 'epoch {:03d}/{:03d}, [{:03d}/{:03d}] | base_loss: {:6f} | contra_loss: {:6f} | psnr: {:4f}, time: {:3f}s'.format(epoch, self.epochs, iter, len(self.dataloader_train), 
                                    base_loss.item(), contra_loss.item(), psnr_train, toc-tic)
                    print(msg)
                    tic = time.time()
                if step == self.stage1_iters:
                    torch.save({
                        'epoch': epoch,
                        'base_state_dict': self.model.state_dict(),
                        'tran_state_dict': self.feat_extractor.state_dict(),
                        'base_optim': self.base_optimizer.state_dict(),
                        'tran_optim': self.tran_optimizer.state_dict(),
                    }, os.path.join(self.opt.save_path, 'latest_stage1.tar'))
                # save_model
                if step % self.opt.save_freq == 0:
                    torch.save(self.model.state_dict(), os.path.join(self.opt.save_path, 'net_epoch_{}.pth'.format(step)))
                    torch.save(self.feat_extractor.state_dict(), os.path.join(self.opt.save_path, 'feat_epoch_{}.pth'.format(step)))
            torch.save({
                'epoch': epoch,
                'base_state_dict': self.model.state_dict(),
                'tran_state_dict': self.feat_extractor.state_dict(),
                'base_optim': self.base_optimizer.state_dict(),
                'tran_optim': self.tran_optimizer.state_dict(),
            }, os.path.join(self.opt.save_path, 'latest.tar'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DRSformer_train')
    parser.add_argument("--batch_size", type=int, default=4, help="Training batch size")
    parser.add_argument("--tot_iters", type=int, default=400000, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=3e-4, help="initial learning rate")
    parser.add_argument("--save_path", type=str, default="checkpoints/DRSformer-H8L1214spascratch-coic", help='path to save models and log files')
    parser.add_argument("--save_freq", type=int, default=5000, help='save intermediate model')
    parser.add_argument("--use_GPU", action="store_true", help='use GPU or not')
    parser.add_argument("--gpu_id", type=str, default="3", help='GPU id')
    parser.add_argument("--data_paths", type=str, default="datasets/Rain200H/train, \
                       datasets/Rain200L/train/, datasets/Rain800/train/, \
                       datasets/DID/train, datasets/DDN/train, datasets/spa")
    parser.add_argument("--model_name", type=str, default="DRSformer", help="training model name")
    parser.add_argument("--crop_size", type=int, default=128)
    parser.add_argument("--aug_times", type=int, default=1, help="augmentation times")
    parser.add_argument("--num_workers", type=int, default=8, help="number of workers")
    parser.add_argument("--seed", type=int, default=100, help='random seed')
    parser.add_argument("--dim_in", type=int, default=128, help='dimension of code z')
    parser.add_argument("--contra_loss_weight", type=float, default=0.2, help="contra_loss_weight")
    parser.add_argument("--temperature", type=float, default=1.0, help="temperature")
    parser.add_argument("--n_neg", type=int, default=4, help="number of negative examples")
    opt = parser.parse_args()

    dim_type = "_{}d".format(opt.dim_in)
    contra_type = "_{}contra".format(opt.contra_loss_weight)
    opt.save_path = opt.save_path + dim_type + contra_type

    os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.gpu_id)    
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    exp = Experiments(opt=opt)
    exp.train()
