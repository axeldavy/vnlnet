"""
Copyright (C) 2018  Axel Davy

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
"""

import os
import os.path
import argparse
import time
import numpy as np
import fnmatch
import errno
import cv2
import torch
import torch.nn as nn
from torch.autograd import Variable
from models import ModifiedDnCNN
from video_patch_search import VideoPatchSearch
from dataset import rgb_to_gray

import imageio
import tifffile

def mkdir_p(path):
    """
    Create a directory without complaining if it already exists.
    """
    if path:
        try:
            os.makedirs(path)
        except OSError as exc: # requires Python > 2.5
            if exc.errno == errno.EEXIST and os.path.isdir(path):
                pass
            else: raise

def get_files_pattern(d, pattern):
    """
    List elements in the directory d with pattern.
    Sort the elements.
    """
    files = os.listdir(d)
    files = fnmatch.filter(files, pattern)
    return sorted(files)

def load_file(f, gray):
    """
    Load a file f.
    gray: whether the image
    should be gray
    """
    if f[-4:] == 'tiff' or f[-3:] == 'tif':
        img = tifffile.imread(f)
    else:
        img = imageio.imread(f)
    if len(img.shape) == 2:
        img = np.expand_dims(img, 2)
    img = np.asarray(img, dtype=np.float32)
    if gray and img.shape[2] > 1:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = np.expand_dims(img, 2)
    assert(img.shape[2] == 1 or img.shape[2] == 3) # Gray or RGB. Please convert RGBA to RGB.
    img = np.asarray(img, dtype=np.float32)
    img = img/255.

    return img

def write_file(f, img):
    """
    Write a file f.
    """
    img = np.squeeze(img)
    if f[-4:] == 'tiff' or f[-3:] == 'tif':
        tifffile.imsave(f, img)
    else:
        img = np.floor(img + 0.5)
        img[img < 0] = 0
        img[img > 255] = 255
        img = np.asarray(img, dtype=np.uint8)
        imageio.imwrite(f, img)

def load_files_with_pattern(dpattern, first, last, gray):
    files = [dpattern % (i) for i in range(first, last+1)]
    images = []
    for f in files:
        img = load_file(f, gray)
        images.append(img)
    return np.ascontiguousarray(images)

def load_files_at_dir(d, pattern, gray):
    files = get_files_pattern(d, pattern)
    images = []
    for f in files:
        img = load_file(os.path.join(d, f), gray)
        images.append(img)
    return np.ascontiguousarray(images), files

def load_net(net_path, cuda):
    if not(os.path.isfile(net_path)):
        assert(False)
    if cuda:
        data_net = torch.load(net_path)
    else:
        data_net = torch.load(net_path, map_location='cpu')
    return data_net['args'], data_net['model_state_dict']

def test(training_args, model_state_dict, video, foutnames, only_frame, cuda, add_noise):
    # non-random noise generation for reproducible tests
    np.random.seed(2018)

    print ("Training args:\n")
    print (training_args)

    model = ModifiedDnCNN(input_channels=training_args.input_channels,
                          output_channels=training_args.output_channels,
                          nlconv_features=training_args.nlconv_features,
                          nlconv_layers=training_args.nlconv_layers,
                          dnnconv_features=training_args.dnnconv_features,
                          dnnconv_layers=training_args.dnnconv_layers)

    if cuda:
        device = torch.device("cuda:0")
        model.to(device)

    model.load_state_dict(model_state_dict)
    model.eval()

    ps = VideoPatchSearch(patch_search_width=training_args.nn_patch_width, patch_data_width=1,
                          input_dtype=np.float32, past_frames=training_args.past_frames,
                          future_frames=training_args.future_frames,
                          search_width=training_args.search_window_width)

    if add_noise:
        video_noised = video + training_args.sigma * np.random.randn(video.shape[0], video.shape[1], video.shape[2], video.shape[3])
    else:
        video_noised = video

    # Add a black border spatiotemporally
    video_noised_bigger = np.zeros([video.shape[0] + training_args.past_frames + training_args.future_frames, video.shape[1] + training_args.nn_patch_width-1, video.shape[2] + training_args.nn_patch_width-1, video.shape[3]], dtype=np.float32)
    border_size = training_args.nn_patch_width//2
    temporal_slice = slice(training_args.past_frames, video_noised_bigger.shape[0]-training_args.future_frames)

    video_noised_bigger[temporal_slice, border_size:-border_size, border_size:-border_size, :] = video_noised[:,:,:,:]

    # Replace black frames with future (or past frames)
    # Make it so different frames are seen during denoising
    for i in range(training_args.past_frames):
        video_noised_bigger[i, border_size:-border_size, border_size:-border_size, :] = video_noised_bigger[training_args.past_frames+training_args.future_frames+1+i, border_size:-border_size, border_size:-border_size, :]
    for i in range(training_args.future_frames):
        video_noised_bigger[-i, border_size:-border_size, border_size:-border_size, :] = video_noised_bigger[-(training_args.past_frames+training_args.future_frames+1+i), border_size:-border_size, border_size:-border_size, :]

    if training_args.oracle_mode == 1:
        assert(add_noise)
        video_bigger = np.zeros([video.shape[0] + training_args.past_frames + training_args.future_frames, video.shape[1] + training_args.nn_patch_width-1, video.shape[2] + training_args.nn_patch_width-1, video.shape[3]], dtype=np.float32)
        video_bigger[temporal_slice, border_size:-border_size, border_size:-border_size, :] = video[:,:,:,:]

    video_search_gray = rgb_to_gray(video_bigger if training_args.oracle_mode == 1 else video_noised_bigger)
    if len(video_search_gray.shape) == 3:
        video_search_gray = video_search_gray[:,:,:,np.newaxis]

    for i in range(video.shape[0]):
        print (foutnames[i])
        if only_frame >= 0 and i != only_frame:
            continue
        video_extract = video_noised_bigger[i:(i+training_args.past_frames+1+training_args.future_frames),:,:,:]
        video_search_extract_gray = video_search_gray[i:(i+training_args.past_frames+1+training_args.future_frames),:,:,:]
        img_noised = video_noised[i, :, :, :]

        nearest_neighbors_indices = ps.compute(video_search_extract_gray, training_args.past_frames)

        if training_args.pass_nn_value:
            img_noised_patch_stack = ps.build_neighbors_array(video_extract, nearest_neighbors_indices[:-(2*border_size),:-(2*border_size),:])
        else:
            img_noised_patch_stack = img_noised

        img_noised = img_noised.transpose(2, 0, 1)
        img_noised_patch_stack = img_noised_patch_stack.transpose(2, 0, 1)

        img_noised = np.expand_dims(img_noised, 0)
        img_noised = torch.Tensor(img_noised)
        img_noised_patch_stack = np.expand_dims(img_noised_patch_stack, 0)
        img_noised_patch_stack = torch.Tensor(img_noised_patch_stack)

        with torch.no_grad():
            dtype = torch.cuda.FloatTensor if cuda else torch.FloatTensor
            img_noised = Variable(img_noised.type(dtype))
            img_noised_patch_stack = Variable(img_noised_patch_stack.type(dtype))

            im_noise_estim = model(img_noised_patch_stack)
            outim = torch.clamp(img_noised-im_noise_estim, 0., 1.)

            dirname = os.path.dirname(foutnames[i])
            mkdir_p(dirname)
            outimg = outim.data.cpu().numpy()[0]
            outimg = 255 * outimg.transpose(1, 2, 0)
            write_file(foutnames[i], outimg)

            """
            # If one needs the noisy frames, uncomment
            if add_noise:
                mkdir_p('test_input')
                write_file('test_input/' + files[i], 255. * video[i, :, :, :])
                mkdir_p('test_input_noisy')
                img_noised = img_noised.data.cpu().numpy()[0]
                img_noised = 255 * img_noised.transpose(1, 2, 0)
                write_file('test_input_noisy/' + files[i], img_noised)
                write_file('test_input_noisy/' + files[i][:-3]+'tiff', img_noised)
            """

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=('Testing VNLnet'))
    subparsers = parser.add_subparsers(dest='cmd', help='directory or pattern',
                                       metavar='{directory, pattern}')

    # Parser for the "directory" command
    parser_dir = subparsers.add_parser('directory', help='Denoise a video sequence in a directory')
    parser_dir.add_argument('--net', type=str, default='mynetwork/net.pth', help='Path to the network')
    parser_dir.add_argument('--input', type=str, default="", help='Path to input directory')
    parser_dir.add_argument('--output', type=str, default="", help='Path to output directory')
    parser_dir.add_argument('--only_frame', type=int, default=-1, help='Only denoise the frame at this index (start from 0)')
    parser_dir.add_argument('--add-noise', action='store_true', help='Add synthetic noise (for the trained sigma)')
    parser_dir.add_argument('--cpu', action='store_true', help="Run model on CPU")

    # Parser for the "pattern" command
    parser_pat = subparsers.add_parser('pattern', help='Denoise a video sequence with a pattern (like input_%3d.tiff)')
    parser_pat.add_argument('--net', type=str, default='mynetwork/net.pth', help='Path to the network')
    parser_pat.add_argument("--input_pattern", type=str, default='', help='Pattern for input files')
    parser_pat.add_argument("--output_pattern", type=str, default='', help='Pattern for output files')
    parser_pat.add_argument("--first", type=str, default='', help='Index of the first frame')
    parser_pat.add_argument("--last", type=str, default='', help='Index of the last frame')
    parser_pat.add_argument('--add-noise', action='store_true', help='Add synthetic noise (for the trained sigma)')
    parser_pat.add_argument('--cpu', action='store_true', help="Run model on CPU")

    args = parser.parse_args()

    cuda = not args.cpu and torch.cuda.is_available()
    training_args, model_state_dict = load_net(args.net, cuda)

    if args.cmd == 'directory':
        video, files = load_files_at_dir(args.input, '*.png', not(training_args.color))
        foutnames = [os.path.join(args.output, f) for f in files]
        only_frame = int(args.only_frame)
    else:
        video = load_files_with_pattern(args.input_pattern, int(args.first), int(args.last), not(training_args.color))
        foutnames = [args.output_pattern % (i) for i in range(int(args.first), int(args.last)+1)]
        only_frame = -1

    if video.shape[0] == 0:
        print ('No file found')
        assert(False)

    test(training_args, model_state_dict, video, foutnames, only_frame, cuda, args.add_noise)
