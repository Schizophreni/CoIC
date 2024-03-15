from models.select_model import define_model
import argparse
from thop import profile
from thop import clever_format
import torch
import numpy as np
import os
from tqdm import tqdm
from models.contrastive import FeatureExtract


os.environ["CUDA_VISIBLE_DEVICES"] = "5"

args = argparse.ArgumentParser()
args = args.parse_args()
args.use_GPU = True
args.dim_in = 128

enc = FeatureExtract(3, 32).cuda()

def obtain_model(model_name):
    args.model_name = model_name
    if model_name == "BRN":
        args.inter_iter = 8
    elif model_name == "RCDNet":
        args.num_map = 32
        args.num_channel = 32
        args.inter_iter = 17
        args.num_block = 4
    elif model_name == "DGUNet":
        args.depth = 5
    return define_model(args)

x = torch.rand(1, 3, 512, 512).cuda()
x_r = torch.rand(1, 3, 128, 128).cuda()

model = obtain_model("DRSformer").cuda()
test_type = "tran"
print("... warm up")
with torch.no_grad():
    for _ in range(20):
        model(x, None, "normal")

torch.cuda.synchronize()
starter = torch.cuda.Event(enable_timing=True)
ender = torch.cuda.Event(enable_timing=True)
timings = np.zeros((30,))
print("... testing")
with torch.no_grad():
    for rep in tqdm(range(30)):
        starter.record()
        if test_type == "normal":
            model(x, None, "normal")
        else:
            z, _ = enc(x_r)
            model(x, z, "tran")
        ender.record()
        torch.cuda.synchronize()
        timings[rep] = starter.elapsed_time(ender)
print(timings)
print("Avg: {}, std: {}".format(timings.mean(), timings.std()))
