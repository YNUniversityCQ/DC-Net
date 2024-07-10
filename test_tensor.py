import os
import cv2
import time
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch.nn as nn
from lib import loaders, modules
from torchmetrics import R2Score
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda:0")
model = modules.DeepAE()
model.load_state_dict(torch.load('model_result/main_model.pt'))
model.to(device)

def combine_loss(pred, target):
    criterion = nn.MSELoss()
    loss = criterion(pred, target)
    return loss

def main_worker():

    # 加载测试数据
    test_data = loaders.DC_Net(phase='test')
    test_dataloader = DataLoader(test_data, shuffle=False, pin_memory=True, batch_size=1, num_workers=4)

    interation = 0
    MSE = []
    for sample, mask, target, img_name in tqdm(test_dataloader):
        interation += 1

        sample, mask, target = sample.cuda(), mask.cuda(), target.cuda()

        with torch.no_grad():
            pre = model(sample, mask)
            loss = combine_loss(pre, target)
        MSE.append(loss)

    mse_err = sum(MSE) / len(MSE)

    print('测试集平均绝对误差：', mse_err)

if __name__ == '__main__':
 main_worker()