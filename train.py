"""
Copyright (C) 2018  Axel Davy
Copyright (C) 2018  Yiqi Yan

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.


This is a modified version derived from
https://github.com/SaoYan/DnCNN-PyTorch/blob/master/train.py
https://github.com/SaoYan/DnCNN-PyTorch/blob/master/utils.py
"""

import os
import argparse
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.utils as utils
from tensorboardX import SummaryWriter
from models import ModifiedDnCNN
from dataset import Dataset
from skimage.measure.simple_metrics import compare_psnr

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(mean=0, std=math.sqrt(2./9./64.)).clamp_(-0.025,0.025)
        nn.init.constant(m.bias.data, 0.0)

def batch_PSNR(img, imclean, data_range):
    Img = img.data.cpu().numpy().astype(np.float32)
    Iclean = imclean.data.cpu().numpy().astype(np.float32)
    PSNR = 0
    for i in range(Img.shape[0]):
        PSNR += compare_psnr(Iclean[i,:,:,:], Img[i,:,:,:], data_range=data_range)
    return (PSNR/Img.shape[0])

def main(args):
    # Load dataset
    print('> Loading dataset ...')
    dataset_train = Dataset(args.train_dir, color_mode=args.color, sigma=args.sigma,
                            oracle_mode=args.oracle_mode, past_frames=args.past_frames,
                            future_frames=args.future_frames,
                            search_window_width=args.search_window_width, nn_patch_width=args.nn_patch_width,
                            pass_nn_value=args.pass_nn_value)
    dataset_val = Dataset(args.val_dir, color_mode=args.color, sigma=args.sigma,
                          oracle_mode=args.oracle_mode, past_frames=args.past_frames,
                          future_frames=args.future_frames,
                          search_window_width=args.search_window_width, nn_patch_width=args.nn_patch_width,
                          pass_nn_value=args.pass_nn_value, patch_stride=20)
    loader_train = DataLoader(dataset=dataset_train, num_workers=2, \
                              batch_size=args.batch_size, shuffle=True)
    print('\t# of training samples: %d\n' % int(len(dataset_train)))

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    writer = SummaryWriter(args.save_dir)

    # Create model
    args.input_channels = dataset_train.data_num_channels()
    args.output_channels = (3 if args.color else 1)
    args.nlconv_features = 32
    args.nlconv_layers = 4
    args.dnnconv_features = (96 if args.color else 64) # Just like FFDNet
    args.dnnconv_layers = (12 if args.color else 15) # Just like FFDNet
    model = ModifiedDnCNN(input_channels=args.input_channels,
                          output_channels=args.output_channels,
                          nlconv_features=args.nlconv_features,
                          nlconv_layers=args.nlconv_layers,
                          dnnconv_features=args.dnnconv_features,
                          dnnconv_layers=args.dnnconv_layers)

    model.apply(weights_init_kaiming)
    criterion = nn.MSELoss(size_average=False)

    # Move to GPU
    device = torch.device("cuda:0")
    model.to(device)
    criterion.cuda()

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    step = 0
    current_lr = args.lr

    # Training
    for epoch in range(0, args.epochs):
        if (epoch+1) >= args.milestone[1]:
            current_lr = args.lr / 1000.
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr
        elif (epoch+1) >= args.milestone[0]:
            current_lr = args.lr / 10.
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr

        # train over all data in the epoch
        for i, data in enumerate(loader_train, 0):
            # Pre-training step
            model.train()
            model.zero_grad()
            optimizer.zero_grad()

            (imgn_stack_train, imgn_train, img_train) = data

            img_train = Variable(img_train.cuda(), volatile=True)
            imgn_train = Variable(imgn_train.cuda(), volatile=True)
            imgn_stack_train = Variable(imgn_stack_train.cuda(), volatile=True)

            # Evaluate model and optimize it
            out_train = model(imgn_stack_train)
            loss = criterion(out_train, imgn_train-img_train) / (img_train.size()[0]*2)
            loss.backward()
            optimizer.step()

            if step % 10 == 0:
                # Results
                model.eval()
                img_predict = torch.clamp(imgn_train-out_train, 0., 1.)
                psnr_train = batch_PSNR(img_predict, img_train, 1.)
                print('[epoch %d][%d/%d] loss: %.4f PSNR_train: %.4f' %\
                    (epoch+1, i+1, len(loader_train), loss.item(), psnr_train))
                # Log the scalar values
                writer.add_scalar('loss', loss.item(), step)
                writer.add_scalar('PSNR on training data', psnr_train, step)
            step += 1

        model.eval()
        psnr_val = 0
        for valimg in dataset_val:
            with torch.no_grad():
                (imgn_stack_val, imgn_val, img_val) = valimg
                imgn_stack_val = torch.unsqueeze(imgn_stack_val, 0)
                imgn_val = torch.unsqueeze(imgn_val, 0)
                img_val = torch.unsqueeze(img_val, 0)
                img_val, imgn_val = Variable(img_val.cuda()), Variable(imgn_val.cuda())
                imgn_stack_val = Variable(imgn_stack_val.cuda())
                out_val = model(imgn_stack_val)
                img_predict = torch.clamp(imgn_val-out_val, 0., 1.)
                psnr_val += batch_PSNR(img_predict, img_val, 1.)
        psnr_val /= len(dataset_val)
        print('\n[epoch %d] PSNR_val: %.4f' % (epoch+1, psnr_val))
        writer.add_scalar('PSNR on validation data', psnr_val, epoch)
        writer.add_scalar('Learning rate', current_lr, epoch)

        net_data = { \
            'model_state_dict': model.state_dict(), \
            'args': args\
        }
        torch.save(net_data, os.path.join(args.save_dir, 'net.pth'))

        # Prepare next epoch
        dataset_train.prepare_epoch()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='VNLnet Training')
    parser.add_argument('--train_dir', type=str, default='train/', help='Path containing the training data')
    parser.add_argument('--val_dir', type=str, default='val/', help='Path containing the validation data')
    parser.add_argument('--save_dir', type=str, default='mynetwork', help='Path to store the logs and the network')
    parser.add_argument('--batch_size', type=int, default=128, help='Training batch size')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--milestone', nargs=2, type=int, default=[12, 17], help='When to decay learning rate')
    parser.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate')
    parser.add_argument('--sigma', type=float, default=20, help='Simulated noise level')
    parser.add_argument('--color', action='store_true', help='Train with color instead of grayscale')
    parser.add_argument('--oracle_mode', type=int, default=0, help='Oracle mode (0: no oracle, 1: image ground truth)')
    parser.add_argument('--past_frames',  type=int, default=7, help='Number of past frames')
    parser.add_argument('--future_frames',  type=int, default=7, help='Number of future frames')
    parser.add_argument('--search_window_width',  type=int, default=41,  help='Search window width for the matches')
    parser.add_argument('--nn_patch_width',  type=int, default=41, help='Width of the patches for matching')
    parser.add_argument('--pass_nn_value', action='store_true', \
                        help='Whether to pass the center pixel value of the matches (noisy image)')
    args = parser.parse_args()

    # The images are normalized
    args.sigma /= 255.

    print(args)

    main(args)
