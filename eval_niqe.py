import pyiqa
import argparse
import torch
import glob
import os
import numpy as np


args = argparse.ArgumentParser()
args.add_argument("--data_dir", type=str, default=".", help="image dir")
opt = args.parse_args()

iqa_metric = pyiqa.create_metric('niqe', device=torch.device('cpu'), as_loss=False)
imgs = glob.glob(os.path.join(opt.data_dir, '*.png'))
metrics = []
for img in imgs:
    score = iqa_metric(img)
    print('{}: {}'.format(img, score))
    metrics.append(score)
mean_score = np.array(metrics).mean()

print('Test path: {}, Avg: {}'.format(opt.data_dir, mean_score))
