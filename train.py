from __future__ import print_function, division
import os
import time
import copy
import torch
import warnings
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
from lib import loaders, modules
from torchsummary import summary
from collections import defaultdict
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader


if __name__ == '__main__':
    print('GPU:', torch.cuda.device_count())
    device = torch.device("cuda:0")

    # ignore warning
    warnings.filterwarnings("ignore")

    # speed
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"

    Radio_train = loaders.DC_Net(phase="train")
    Radio_val = loaders.DC_Net(phase="val")

    image_datasets = {
        'train': Radio_train, 'val': Radio_val
    }

    main_dataloaders = {
        'train': DataLoader(Radio_train, batch_size=4, shuffle=True, num_workers=4, pin_memory=True),
        'val': DataLoader(Radio_val, batch_size=4, shuffle=True, num_workers=4, pin_memory=True)
    }

    # loading network parameter
    torch.set_default_dtype(torch.float32)
    torch.set_default_tensor_type('torch.FloatTensor')
    torch.backends.cudnn.enabled
    model = modules.DeepAE()
    model.cuda()
    summary(model, input_size=[(1, 256, 256),
                               (1, 256, 256)])

    def combine_loss(pred, target, metrics):
        criterion = nn.MSELoss()
        loss = criterion(pred, target)
        metrics['MSE_loss'] += loss.data.cpu().numpy() * target.size(0)

        return loss

    def print_metrics(metrics, epoch_samples, phase):
        outputs = []
        for k in metrics.keys():
            outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))

        print("{}: {}".format(phase, ", ".join(outputs)))

    def train_model(model, optimizer, scheduler, num_epochs=1, targetType="dense"):
        best_model_wts = copy.deepcopy(model.state_dict())
        best_loss = 1e10

        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)

            since = time.time()

            # 训练并验证
            for phase in ['train', 'val']:
                if phase == 'train':
                    scheduler.step()
                    for param_group in optimizer.param_groups:
                        print("learning rate", param_group['lr'])

                    model.train()
                else:
                    model.eval()

                metrics = defaultdict(float)
                epoch_samples = 0
                if targetType == "dense":
                    for sample, mask, target, name in tqdm(main_dataloaders[phase]):
                        sample, mask, target = sample.to(device), mask.to(device), target.to(device)

                        optimizer.zero_grad()

                        with torch.set_grad_enabled(phase == 'train'):

                            outputs = model(sample, mask)

                            loss = combine_loss(outputs, target, metrics)

                            if phase == 'train':
                                loss.backward()
                                optimizer.step()

                        epoch_samples += target.size(0)

                print_metrics(metrics, epoch_samples, phase)
                epoch_loss = metrics['MSE_loss'] / epoch_samples

                if phase == 'val' and epoch_loss < best_loss:
                    print("saving best model")
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())

            time_elapsed = time.time() - since
            print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

        print('Best val loss: {:4f}'.format(best_loss))

        # saving best weight
        model.load_state_dict(best_model_wts)
        return model

    # conduct training
    optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=30, gamma=0.1)
    model = train_model(model, optimizer_ft, exp_lr_scheduler)

    # build the dir
    try:
        os.mkdir('model_result')
    except OSError as error:
        print(error)

    # saving
    torch.save(model.state_dict(), 'model_result/main_model.pt')