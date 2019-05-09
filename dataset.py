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
import numpy as np
import numpy.random
import cv2
import torch
import torch.utils.data as udata
import random

from video_patch_search import VideoPatchSearch

# Using the same weights for each channel
# gives optimal contrast over noise.
def rgb_to_gray(img):
    img = np.asarray(img, dtype=np.float32)
    if (len(img.shape) == 3 or img.shape[3] == 1):
        return img
    res = np.dot(img[...,:3], [0.57735, 0.57735, 0.57735])
    return np.asarray(res, dtype=np.float32)

class Dataset(udata.Dataset):
    def __init__(self, data_path, color_mode=False, sigma=25, oracle_mode=0, past_frames=7, future_frames=7, search_window_width=41, nn_patch_width=41, pass_nn_value=False, patch_width=44, patch_stride=5):
        super(Dataset, self).__init__()
        self.color = color_mode
        self.sigma = sigma

        categories = os.scandir(data_path)
        categories = [c for c in categories if not c.name.startswith('.') and c.is_dir()]

        video_paths_dict = {}
        video_paths_list = []
        for c in categories:
            list_for_category = []
            paths = os.scandir(c.path)
            paths = [p for p in paths if not p.name.startswith('.') and p.is_dir()]
            video_paths_list.extend(paths)
            video_paths_dict[c.name] = [p.path for p in paths]
        categories = [c.name for c in categories]

        print ('%d categories' % len(categories))
        print ('%d videos' % len(video_paths_list))

        self.categories = categories
        self.video_paths_dict = video_paths_dict

        self.patch_width_nn = nn_patch_width
        self.patch_data_width_nn = 1
        self.past_frames = past_frames
        self.future_frames = future_frames
        self.num_neighbors = 1 + past_frames + future_frames
        self.patch_width = patch_width
        self.patch_stride = patch_stride
        self.oracle_mode = oracle_mode
        self.pass_nn_value = pass_nn_value

        self.ps = VideoPatchSearch(patch_search_width=self.patch_width_nn, patch_data_width=self.patch_data_width_nn,
                                   past_frames=self.past_frames, future_frames=self.future_frames,
                                   input_dtype=np.float32,
                                   search_width=search_window_width)


        np.random.seed(2018)
        self.prepare_epoch()


    def prepare_epoch(self):
        """
        Read random videos of the database
        """
        subdirnames = ['/01/', '/02/', '/03/', '/04/', '/05/', '/06/', '/07/', '/08/', '/09/']
        filenames= ['001.png', '002.png', '003.png', '004.png', '005.png', '006.png', '007.png', '008.png', '009.png', '010.png', '011.png', '012.png', '013.png', '014.png', '015.png']
        burst_nums = np.random.randint(len(subdirnames), size=len(self.categories*10))
        frame_nums = np.random.randint(self.past_frames, high=len(filenames)-self.future_frames, size=len(self.categories*10))
        i = 0
        self.videos = []
        self.keys = []
        for c in self.categories:
            paths = self.video_paths_dict[c]
            paths = np.random.permutation(paths)
            for p in paths[:3]:
                dir_path = p + subdirnames[burst_nums[i]]
                if not (os.path.exists(dir_path)):
                    continue
                video = []
                for f in range(frame_nums[i]-self.past_frames, frame_nums[i]+self.future_frames+1):
                    img = cv2.imread(dir_path + filenames[f])
                    img = np.asarray(img, dtype=np.float32)
                    if self.color:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    else:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        img = np.expand_dims(img, 2)
                    img = np.asarray(img, dtype=np.float32)
                    img = img/255.
                    video.append(img)
                ref_image = video[self.past_frames]
                video = np.asarray(video, dtype=np.float32)

                video_noised = video + self.sigma * np.random.randn(video.shape[0], video.shape[1], video.shape[2], video.shape[3])
                video_noised = np.asarray(video_noised, dtype=np.float32)

                video_search_gray = rgb_to_gray(video if self.oracle_mode == 1 else video_noised)

                nn = self.ps.compute(video_search_gray, self.past_frames)

                self.videos.append((video_noised[self.past_frames, ...] - ref_image, video_noised, nn))
                ys = range(2*self.patch_width, ref_image.shape[0]-2*self.patch_width, self.patch_stride)
                xs = range(2*self.patch_width, ref_image.shape[1]-2*self.patch_width, self.patch_stride)
                xx, yy = np.meshgrid(xs, ys)
                xx = np.asarray(xx.flatten(), dtype=np.uint32)
                yy = np.asarray(yy.flatten(), dtype=np.uint32)
                self.keys.append(np.stack([i*np.ones([len(xx)], dtype=np.uint32), xx, yy]).T)

                i = i + 1
        self.keys = np.concatenate(self.keys, axis=0)
        self.num_keys = self.keys.shape[0]
        self.indices = [i for i in range(self.num_keys)]

        random.shuffle(self.indices)


    def __len__(self):
        return self.num_keys

    def data_num_channels(self):
        return (3 if self.color else 1) * (self.num_neighbors if self.pass_nn_value else 1)

    def __getitem__(self, index):
        key = self.keys[self.indices[index],:]
        patch_width = self.patch_width
        i = key[0]
        x = key[1]
        y = key[2]
        anchor = self.patch_width_nn//2
        noise = self.videos[i][0][(y-patch_width):y, (x-patch_width):x,:]

        if self.pass_nn_value:
            nn_patch = self.videos[i][2][(y-anchor-patch_width):(y-anchor), (x-anchor-patch_width):(x-anchor),:]
            patch_stack = self.ps.build_neighbors_array(self.videos[i][1], nn_patch)
        else:
            patch_stack = self.videos[i][1][self.past_frames, (y-patch_width):y, (x-patch_width):x,:]

        patch_stack = patch_stack.transpose(2, 0, 1)
        noise = noise.transpose(2, 0, 1)

        return (torch.Tensor(patch_stack), torch.Tensor(noise))
