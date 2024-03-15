import os
import torch
from tqdm import tqdm
import time

# declare which gpu device to use
cuda_device = '0'

def check_mem(cuda_device):
    devices_info = os.popen('"/usr/bin/nvidia-smi" --query-gpu=memory.total,memory.used --format=csv,nounits,noheader').read().strip().split("\n")
    total, used = devices_info[int(cuda_device)].split(',')
    return total,used

def occumpy_mem(cuda_device, block_mem):
    total, used = check_mem(cuda_device)
    print(total, used)
    total = int(total)
    used = int(used)
    max_mem = int(total * 0.9)
    # block_mem = max_mem - used
    x = torch.cuda.FloatTensor(256,1024,block_mem)
    del x
    
if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device
    occumpy_mem(cuda_device, block_mem=1024 * 14)
    while True:
        time.sleep(1)
