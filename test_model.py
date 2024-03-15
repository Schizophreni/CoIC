from PIL import Image
import cv2
import os
import argparse
import numpy as np
import torch
from evaluation import rgb_to_y
from evaluation import psnr as compare_psnr
from evaluation import ssim as compare_ssim
from models.select_model import define_model
from models.contrastive import MoCo
import torch.nn.functional as F
import torchvision as TV
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
from tqdm import tqdm
import glob


parser = argparse.ArgumentParser(description="Mix_test")
parser.add_argument("--save_path", type=str, default="results", help='path to save results')
parser.add_argument("--use_GPU", action="store_true", help='use GPU or not')
parser.add_argument("--gpu_id", type=str, default="0", help='GPU id')
parser.add_argument("--inter_iter", type=int, default=6, help='number of recursive stages')
parser.add_argument("--test_model", type=str,default='')
parser.add_argument("--feat_ext", type=str, default=".", help="checkpoint of encoder")
parser.add_argument("--model_name", type=str, default="DGUNet", help="model name")
parser.add_argument("--testset", type=str, default="Rain100L", help="test dataset")
parser.add_argument("--load_mode", type=str, default="normal", help="process mode")
parser.add_argument("--tile", type=int, default=512, help="size of tile")
parser.add_argument("--tile_overlap", type=int, default=0, help="overlapping of different tiles")
opt = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_id
device = torch.device('cuda') if opt.use_GPU else torch.device("cpu")

def obtain_crops(base_h, base_w, H, W, stride):
    stride_h, stride_w = stride, stride
    nh, nw = (H-base_h) // stride_h + 1, (W-base_w) // stride_w + 1
    hs, ws = [i*stride_h for i in range(nh)], [j*stride_w for j in range(nw)]
    if (H - base_h) % stride_h != 0:
        hs.append(H - base_h)
    if (W - base_w) % stride_w != 0:
        ws.append(W - base_w)
    return hs, ws

def obtain_score_map(base_h, base_w):
    ## obtain score map, ref: https://github.com/jiexiaou/IDT/blob/main/test_full_size.py
    hs, hw = torch.arange(base_h), torch.arange(base_w)
    hs = (hs - base_h / 2).unsqueeze(1)  # [H, 1]
    hw = (hw - base_w / 2).unsqueeze(0)  # [1, W]
    scores = 1.0 / torch.sqrt((hs**2 + hw**2 + 1e-3)).float()
    # scores = torch.ones(base_h, base_w).float()
    return scores[None, :, :]

def test_Rain200H():
    input_path = os.path.join('/home1/zhangsy/rh/data/derain/Rain200H/test/rain/X2')
    target_path = os.path.join('/home1/zhangsy/rh/data/derain/Rain200H/test/norain')
    imgs = []
    gts = []
    for i in range(200):
        target_file = "norain-%d.png" % (i + 1)
        input_file = "norain-%dx2.png" %(i + 1)
        imgs.append(os.path.join(input_path, input_file))
        gts.append(os.path.join(target_path, target_file))
    print("process Rain200H! total length: ", len(imgs))
    return imgs, gts

def test_Rain200L():
    input_path = os.path.join('/home1/zhangsy/rh/data/derain/Rain200L/test/rain/X2')
    target_path = os.path.join('/home1/zhangsy/rh/data/derain/Rain200L/test/norain')
    imgs = []
    gts = []
    for i in range(200):
        target_file = "norain-%d.png" % (i + 1)
        input_file = "norain-%dx2.png" %(i + 1)
        imgs.append(os.path.join(input_path, input_file))
        gts.append(os.path.join(target_path, target_file))
    print("process Rain200L! total length: ", len(imgs))
    return imgs, gts

def test_Rain800():
    input_path = os.path.join('/home1/zhangsy/rh/data/derain/Rain800/test/rain')
    target_path = os.path.join('/home1/zhangsy/rh/data/derain/Rain800/test/norain')
    imgs = []
    gts = []
    for i in range(100):
        target_file = "norain-%03d.png" % (i + 1)
        input_file = "rain-%03d.png" %(i + 1)
        imgs.append(os.path.join(input_path, input_file))
        gts.append(os.path.join(target_path, target_file))
    print("process Rain800! total length: ", len(imgs))
    return imgs, gts 

def test_Rain1200():
    input_path = os.path.join('/home1/zhangsy/rh/data/derain/Rain1200_new/test')
    imgs = []
    for i in range(1200):
        input_file = "%d.jpg" % (i+1)
        imgs.append(os.path.join(input_path, input_file))
    return imgs, imgs

def test_Rain1400():
    input_path = os.path.join('/home1/zhangsy/rh/data/derain/Rain14000/test/rain')
    target_path = os.path.join('/home1/zhangsy/rh/data/derain/Rain14000/test/norain')
    imgs, gts = [], []
    for i in range(900, 1000):
        target_file = "%d.jpg" % (i + 1)
        for j in range(14):
            input_file = "%d_%d.jpg" % (i + 1, j + 1)
            imgs.append(os.path.join(input_path, input_file))
            gts.append(os.path.join(target_path, target_file))
    return imgs, gts

def test_Rain12():
    input_path = os.path.join("/home/rw/Public/datasets/derain/Rain12/")
    imgs, gts = [], []
    for i in range(12):
        input_file = "%03d_in.png" % (i+1)
        target_file = "%03d_GT.png" % (i+1)
        imgs.append(os.path.join(input_path, input_file))
        gts.append(os.path.join(input_path, target_file))
    print("process Rain12!, total length: ", len(imgs))
    return imgs, gts

def test_spa():
    input_path = os.path.join('/home1/zhangsy/rh/data/derain/spa/real_test_1000/rain')
    target_path = os.path.join('/home1/zhangsy/rh/data/derain/spa/real_test_1000/gt')
    imgs = []
    gts = []
    for i in range(1000):
        target_file = "%03dgt.png" % (i)
        input_file = "%03d.png" %(i)
        imgs.append(os.path.join(input_path, input_file))
        gts.append(os.path.join(target_path, target_file))
    print("process SPA! total length: ", len(imgs))
    return imgs, gts 

def test_Practical():
    input_path = "/home/rw/Public/datasets/derain/Practical"
    imgs = glob.glob(os.path.join(input_path, "*.jpg"))
    print("Process practical!, total length ", len(imgs))
    return imgs, imgs

def test_RealInt():
    input_path = "/home1/zhangsy/rh/data/derain/Real_Internet"
    imgs = glob.glob(os.path.join(input_path, "*.png"))
    print("Process Real Internet, total length, ", len(imgs))
    return imgs, imgs

def test_eval():
    input_path = "/home/rw/Public/datasets/derain/Real_Internet"
    idxes = [35, 45, 57, 67, 69, 94, 95, 121]
    imgs = []
    for idx in idxes:
        imgs.append(os.path.join(input_path, "rain-%03d.png" % idx))
    return imgs, imgs

def test_Nature():
    input_path = "/home/rw/Public/datasets/derain/test_nature"
    imgs = glob.glob(os.path.join(input_path, "*.jpg"))
    print("Process test nature, total length, ", len(imgs))
    return imgs, imgs

def obtain_test(dataset_name):
    if dataset_name == "Rain200H":
        imgs, gts = test_Rain200H()
    elif dataset_name == "Rain200L":
        imgs, gts = test_Rain200L()
    elif dataset_name == "Rain800":
        imgs, gts = test_Rain800()
    elif dataset_name == "spa":
        imgs, gts = test_spa()
    elif dataset_name == "Rain1200":
        imgs, gts = test_Rain1200()
    elif dataset_name == "Rain1400":
        imgs, gts = test_Rain1400()
    elif dataset_name == "Rain12":
        imgs, gts = test_Rain12()
    elif dataset_name == "Practical":
        imgs, gts = test_Practical()
    elif dataset_name == "Real_internet":
        imgs, gts = test_RealInt()
    elif dataset_name == "Nature":
        imgs, gts = test_Nature()
    elif dataset_name == "test_eval":
        imgs, gts = test_eval()
    return imgs, gts
    
def obtain_model(opt):
    # define model
    print("[===] Build Model")
    opt.dim_in = 128
    if opt.model_name == "BRN":
        opt.inter_iter = 8
    if opt.model_name == "RCDNet":
        opt.num_map, opt.num_block, opt.num_channel, opt.inter_iter = 32, 4, 32, 17
    if opt.model_name == "DGUNet":
        opt.depth = 5
    model = define_model(opt)
    model.to(device)
    if opt.model_name == "DRSformer":
        # model = torch.nn.DataParallel(model)
        pass
    # define feature extractor
    if opt.load_mode == "tran":
        print("[===] Build Encoder")
        encoder = MoCo(in_channels=3, out_channels=32, dim=128, temperature=1.0)
        encoder.to(device)
    # load checkpoint
    if opt.load_mode == "tran":
        encoder.load_state_dict(torch.load(opt.feat_ext))
        encoder.eval()
    else:
        encoder = None
    model.load_state_dict(torch.load(opt.test_model))
    model.eval()
    return model, encoder

def merge_img(base_h, base_w, hs, ws, all_crops, H, W):
    mask = torch.zeros(3, H, W)
    out = torch.zeros(3, H, W)
    all_crops = torch.cat(all_crops, dim=0)
    score_map = obtain_score_map(base_h, base_w)
    cnt = 0
    for h in hs:
        for w in ws:
            out[:, h:h+base_h, w:w+base_w] += all_crops[cnt] * score_map
            mask[:, h:h+base_h, w:w+base_w] += 1.0 * score_map
            cnt += 1
    return out / mask

@torch.no_grad()
def main():
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    tran_transform = transforms.Compose([
        transforms.Resize(size=(128, 128)),  # should divided by 8 (no large than short-length most of images)
        # transforms.ToTensor()
    ])
    os.makedirs(os.path.join(opt.save_path), exist_ok=True)
    psnrs, ssims = [], []
    cnt = 0
    model, encoder = obtain_model(opt)
    imgs, gts = obtain_test(dataset_name=opt.testset)
    save_img = True
    fil = open(os.path.join(opt.save_path, "log.txt"), "a+")
    with tqdm(zip(imgs, gts), ncols=100) as pbar_test: 
        for rain_img, gt_img in tqdm(zip(imgs, gts)):
            # x = Image.open(rain_img).convert("RGB")
            # y = Image.open(gt_img).convert("RGB")
            x = cv2.cvtColor(cv2.imread(rain_img), cv2.COLOR_BGR2RGB)
            y = cv2.cvtColor(cv2.imread(gt_img), cv2.COLOR_BGR2RGB)
            min_h = min(y.shape[0], y.shape[1])
            x, y = transform(x), transform(y)
            if "Nature" in opt.testset or "Rain1200" in opt.testset:
                x, y = torch.chunk(x, chunks=2, dim=-1)        
            H, W = x.shape[-2], x.shape[-1]
            if min(H, W) <= opt.tile:
                x = x.unsqueeze(0).to(device)
                tran_x = tran_transform(x)
                new_H, new_W = ((H + 8)//8)*8, ((W + 8)//8)*8
                padh = new_H - H if H%8!=0 else 0
                padw = new_W - W if W%8!=0 else 0
                x = F.pad(x, (0, padw, 0, padh), 'reflect')
                sh, sw = (H - min_h) // 2, (W-min_h) // 2
                # tran_x = tran_transform(x[:, :, sh:sh+min_h, sw:sw+min_h])
                if opt.load_mode == "tran":
                    z, _ = encoder.encoder_k(tran_x.to(device))
                else:
                    z = None
                if opt.model_name == "DGUNet":
                    outs = model(x, tran_x=z, mode=opt.load_mode)
                    outs = outs[0]
                elif opt.model_name in ["BRN", "IDT", "DRSformer"]:
                    outs = model(x, z, mode=opt.load_mode)
                elif opt.model_name == "RCDNet":
                    outs = model(255.0 * x, z, mode=opt.load_mode)
                    outs = outs[1][-1] / 255.0
                out_img = outs[0, :, :H, :W].cpu().clamp_(0.0, 1.0)
            else:
                tran_x = tran_transform(x)
                hs, ws = obtain_crops(base_h=opt.tile, base_w=opt.tile, H=H, W=W, stride=opt.tile-opt.tile_overlap)
                batch_size = int(6* (512 // opt.tile)**2)
                # print(batch_size)
                batch_cnt = 0  
                x_batch = []
                all_crops = []
                base_h, base_w = opt.tile, opt.tile
                tran_x = tran_x.unsqueeze(0).to(device)
                if opt.load_mode == "tran":
                    z, _ = encoder.encoder_k(tran_x)
                else:
                    z = None
                for start_h in hs:
                    for start_w in ws:
                        batch_cnt += 1
                        crop_x = x[:, start_h:start_h+base_h, start_w:start_w+base_w].unsqueeze(0)
                        x_batch.append(crop_x)
                        cnt += 1
                        if batch_cnt == batch_size or (start_h == hs[-1] and start_w == ws[-1]):
                            crop_xs = torch.cat(x_batch, dim=0).to(device)
                            if opt.load_mode == "tran":
                                zs = z.repeat(batch_cnt, 1)
                            else:
                                zs = None
                            if opt.model_name == "DGUNet":
                                outs = model(crop_xs, tran_x=zs, mode=opt.load_mode)
                                outs = outs[0]
                            elif opt.model_name in ["BRN", "IDT", "DRSformer"]:
                                outs = model(crop_xs, zs, mode=opt.load_mode)
                            elif opt.model_name == "RCDNet":
                                outs = model(255.0 * crop_xs, zs, mode=opt.load_mode)
                                outs = outs[1][-1] / 255.0
                            outs.clamp_(0.0, 1.0)
                            x_batch = []
                            batch_cnt = 0
                            if save_img:
                                all_crops.append(outs.cpu())
                out_img = merge_img(base_h, base_w, hs, ws, all_crops, H, W)
                out_img.clamp_(0.0, 1.0)
            psnr_val = compare_psnr(rgb_to_y(out_img*255.0), rgb_to_y(y*255.0))
            ssim_val = compare_ssim(rgb_to_y(out_img*255.0), rgb_to_y(y*255.0))
            pbar_test.set_postfix(psnr=psnr_val, ssim=ssim_val, H=H, w=W)
            fil.write("{}: {:.4f}, {:.4f}\n".format(rain_img.split("/")[-1], psnr_val, ssim_val))
            psnrs.append(psnr_val)
            ssims.append(ssim_val)

            if save_img:
                TV.utils.save_image(out_img, os.path.join(opt.save_path, rain_img.split("/")[-1]))
    fil.close()


    print('Avg. psnr: ', np.array(psnrs).mean())
    print('Avg. ssim: ', np.array(ssims).mean())
    print('Total patches: ', cnt)

if __name__ == "__main__":
    main()
    a = input("hello ... ")
