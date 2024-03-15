from tensorboardX import SummaryWriter
import argparse
import numpy as np
import os
import torch
from torch.backends import cudnn
import random
import torch.nn.functional as F
from evaluation import psnr as compare_psnr
from functools import reduce
from models.select_model import define_model
from models.contrastive import MoCo
from mix_dataset import getcontrastivemixloader


class CharbonnierLoss(torch.nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
        return loss

class EdgeLoss(torch.nn.Module):
    def __init__(self):
        super(EdgeLoss, self).__init__()
        k = torch.Tensor([[.05, .25, .4, .25, .05]])
        self.kernel = torch.matmul(k.t(),k).unsqueeze(0).repeat(3,1,1,1)
        if torch.cuda.is_available():
            self.kernel = self.kernel.cuda()
        self.loss = CharbonnierLoss()

    def conv_gauss(self, img):
        n_channels, _, kw, kh = self.kernel.shape
        img = F.pad(img, (kw//2, kh//2, kw//2, kh//2), mode='replicate')
        return F.conv2d(img, self.kernel, groups=n_channels)

    def laplacian_kernel(self, current):
        filtered    = self.conv_gauss(current)    # filter
        down        = filtered[:,:,::2,::2]               # downsample
        new_filter  = torch.zeros_like(filtered)
        new_filter[:,:,::2,::2] = down*4                  # upsample
        filtered    = self.conv_gauss(new_filter) # filter
        diff = current - filtered
        return diff

    def forward(self, x, y):
        loss = self.loss(self.laplacian_kernel(x), self.laplacian_kernel(y))
        return loss

class Experiments:
    def __init__(self, opt, n_codes=2):
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
        self.criterion_char = CharbonnierLoss()
        self.criterion_edge = EdgeLoss()
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
        self.base_params = base_parameters
        self.tran_params = tran_parameters
        self.base_optimizer = torch.optim.Adam(params=base_parameters, lr=opt.lr, betas=(0.9, 0.999), eps=1e-8)
        self.tran_optimizer = torch.optim.Adam(params=tran_parameters, lr=opt.lr, betas=(0.9, 0.999), eps=1e-8)
        self.base_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.base_optimizer, opt.epochs, eta_min=1e-6)
        self.tran_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.tran_optimizer, opt.epochs, eta_min=1e-6) 
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
        return int(ckp['epoch']) + 1

    def train(self):
        # Start training
        def compute_loss(restored, gt, w=0.05):
            loss_char = reduce(torch.add, [self.criterion_char(restored[j], gt) for j in range(len(restored))])
            loss_edge = reduce(torch.add, [self.criterion_edge(restored[j], gt) for j in range(len(restored))])
            return loss_char + w*loss_edge

        step = (self.init_epoch-1)*len(self.dataloader_train)
        rain_archives = []  # store recent rain for models with small batch size 
        n_archives = 64 // self.opt.batch_size
        for epoch in range(self.init_epoch, self.epochs + 1):
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
                self.base_optimizer.zero_grad()
                self.tran_optimizer.zero_grad()
                if step <= self.opt.stage1_iters:
                    outs = self.model(img=input_train, tran_x=None, mode="normal")
                    base_loss = compute_loss(outs, target_train)
                    (base_loss).backward()
                    contra_loss = base_loss * 0
                else:
                    logits, labels, z_q = self.feat_extractor(im_q, im_k, im_negs)
                    contra_loss = self.criterion_contra(logits, labels)
                    outs = self.model(img=input_train, tran_x=z_q, mode="tran")
                    base_loss = compute_loss(outs, target_train)
                    contra_loss *= self.opt.contra_loss_weight
                    (base_loss + contra_loss).backward()
                    if outs[0].abs().max() > 1e1:  # may meet invalid batch which results in gradient explosion
                        print('Explode find')
                        torch.nn.utils.clip_grad_norm_(parameters=self.base_params, max_norm=1e-8)
                        torch.nn.utils.clip_grad_norm_(self.tran_params, max_norm=1e-8)
                self.base_optimizer.step()
                self.tran_optimizer.step()

                if (1+iter) % 40 == 0:
                    out_train = torch.clamp(outs[0].detach().cpu(), 0.0, 1.0)
                    psnr_train = compare_psnr(out_train, target_train.cpu(), data_range=1.0)
                    self.writter.add_scalar("base_loss", base_loss.item(), step)
                    self.writter.add_scalar("contra_loss", contra_loss.item(), step)
                    msg = 'epoch {:03d}/{:03d}, [{:03d}/{:03d}] | base_loss: {:6f} | contra_loss: {:6f} | psnr: {:4f}'.format(epoch, self.epochs, iter, len(self.dataloader_train), 
                                    base_loss.item(), contra_loss.item(), psnr_train)
                    print(msg)
                step += 1
                if step == self.opt.stage1_iters:
                    torch.save({
                        'epoch': epoch,
                        'base_state_dict': self.model.state_dict(),
                        'tran_state_dict': self.feat_extractor.state_dict(),
                        'base_optim': self.base_optimizer.state_dict(),
                        'tran_optim': self.tran_optimizer.state_dict(),
                    }, os.path.join(self.opt.save_path, 'latest_stage1.tar'))
            # learning rate scheduler
            self.base_scheduler.step(epoch=epoch)
            self.tran_scheduler.step(epoch=epoch)
            # save_model
            torch.save(self.model.state_dict(), os.path.join(self.opt.save_path, 'net_epoch_{}.pth'.format(epoch)))
            torch.save(self.feat_extractor.state_dict(), os.path.join(self.opt.save_path, 'feat_epoch_{}.pth'.format(epoch)))
            torch.save({
                'epoch': epoch,
                'base_state_dict': self.model.state_dict(),
                'tran_state_dict': self.feat_extractor.state_dict(),
                'base_optim': self.base_optimizer.state_dict(),
                'tran_optim': self.tran_optimizer.state_dict(),
            }, os.path.join(self.opt.save_path, 'latest_{}.tar'.format(epoch)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DGUNet_train')
    parser.add_argument("--depth", type=int, default=5, help="iteration depth")
    parser.add_argument("--batch_size", type=int, default=16, help="Training batch size")
    parser.add_argument("--tot_iters", type=int, default=400000, help="Number of training epochs")
    parser.add_argument("--stage1_iters", type=int, default=40000, help="Number of stage1 training epochs")
    parser.add_argument("--lr", type=float, default=2e-4, help="initial learning rate")
    parser.add_argument("--save_path", type=str, default="logs/DGUNet-H8L1214-coic", help='path to save models and log files')
    parser.add_argument("--save_freq", type=int, default=1, help='save intermediate model')
    parser.add_argument("--use_GPU", action="store_true", help='use GPU or not')
    parser.add_argument("--gpu_id", type=str, default="0", help='GPU id')
    parser.add_argument("--data_paths", type=str, default="/home/wran/Public/datasets/derain/CoIC_datasets/Rain200H/train, \
                       /home/wran/Public/datasets/derain/CoIC_datasets/Rain200L/train/, /home/wran/Public/datasets/derain/CoIC_datasets/Rain800/train/, \
                       /home/wran/Public/datasets/derain/CoIC_datasets/DID/train, /home/wran/Public/datasets/derain/CoIC_datasets/DDN/train")
    parser.add_argument("--model_name", type=str, default="DGUNet", help="training model name")
    parser.add_argument("--crop_size", type=int, default=128)
    parser.add_argument("--aug_times", type=int, default=1, help="augmentation times")
    parser.add_argument("--num_workers", type=int, default=8, help="number of workers")
    parser.add_argument("--seed", type=int, default=1234, help='random seed')
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
