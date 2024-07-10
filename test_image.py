import os
import cv2
import time
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from PIL import Image
from lib import loaders, modules
from torchmetrics import R2Score
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader

# loading model
device = torch.device("cuda:0")
model = modules.DeepAE()
model.load_state_dict(torch.load('model_result/main_model.pt'))
model.to(device)

def main_worker():

    # loading test data
    test_data = loaders.DC_Net(phase='test')
    test_dataloader = DataLoader(test_data, shuffle=False, pin_memory=True, batch_size=1, num_workers=4)

    interation = 0
    err1 = []
    err2 = []
    distance = []
    for sample, mask, target, img_name in test_dataloader:
        interation += 1

        sample, mask, target = sample.cuda(), mask.cuda(), target.cuda()
        # sample = (target * (1 - mask).float()) + mask

        with torch.no_grad():
            pre = model(sample, mask)

        # target
        test1 = torch.tensor([item.cpu().detach().numpy() for item in target]).cuda()
        test1 = test1.squeeze(0)
        test1 = test1.squeeze(0)
        im = test1.cpu().numpy()
        image = im * 255
        images = Image.fromarray(image.astype(np.uint8))

        # predict
        test = torch.tensor([item.cpu().detach().numpy() for item in pre]).cuda()
        test = test.squeeze(0)
        test = test.squeeze(0)
        im1 = test.cpu().numpy()
        predict = im1 * 255
        predict1 = Image.fromarray(predict.astype(np.uint8))


        # # distance
        #
        # arr1 = np.asarray(predict1)
        # arr2 = np.asarray(images)
        #
        # max_sum, max_area, size = -np.inf, None, 4
        #
        # # 数值和最大的 stride×stride 区域
        # for _ in range(254):
        #     for __ in range(254):
        #         kernel = arr1[_:_ + size, __:__ + size]
        #         kernel_sum = np.sum(kernel)
        #         if kernel_sum > max_sum:
        #             max_sum = kernel_sum
        #             max_area = kernel
        #             max_position = (_, __)
        #
        # # 找到最大数值和区域内的最大值
        # area_max = np.argmax(max_area)
        # # 索引最大数值和区域内最大值的坐标
        # max_index = np.unravel_index(area_max, max_area.shape)
        # # 最终的位置索引坐标
        # tx1_index = [max_position[0] + max_index[0], max_position[1] + max_index[1]]
        #
        # # print("原图像最大区域左上角位置:", max_position)
        # # print("Max Region:")
        # # print(max_area)
        # # print("Max Value:", max_index)
        #
        # tx2 = np.argmax(arr2)
        # tx2_index = np.unravel_index(tx2, arr2.shape)
        #
        # distances = np.linalg.norm(np.asarray(tx1_index) - np.asarray(tx2_index))
        #
        # distance.append(distances)

        # calculate rmse
        rmse1 = np.sqrt(np.mean((im - im1) ** 2))
        err1.append(rmse1)
        # calculate nmse
        nmse1 = np.mean((im - im1) ** 2)/np.mean((0 - im) ** 2)
        err2.append(nmse1)

        # 保存
        image_name = os.path.basename(img_name[0]).split('.')[0]
        images.save(os.path.join("image_result", f'{image_name}_target.png'))
        # samples.save(os.path.join("image_result", f'{image_name}_sample.png'))
        predict1.save(os.path.join("image_result", f'{image_name}_predict1.png'))
        print(f'saving to {os.path.join("image_result", image_name)}', "RMSE:", rmse1, "NMSE:", nmse1)

        # the number total of 8000
        if interation >= 8000:
            break
    rmse_err = sum(err1)/len(err1)
    nmse_err = sum(err2) / len(err2)
    # TX_distance = sum(distance) / len(distance)

    print('一阶段测试集均方根误差：', rmse_err)
    print('一阶段测试集归一化均方误差：', nmse_err)
    # print('测试集平均欧氏距离：', TX_distance)


if __name__ == '__main__':
 main_worker()