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

import numpy as np
import pyopencl as cl
import math

import pyximport
pyximport.install(setup_args={"include_dirs":np.get_include()})
from video_patch_search_acc import *

src_type_parameter = {
    np.uint8 : '-DSRC_TYPE=uchar',
    np.int8 : '-DSRC_TYPE=char',
    np.uint16 : '-DSRC_TYPE=ushort',
    np.int16 : '-DSRC_TYPE=short',
    np.uint32 : '-DSRC_TYPE=uint',
    np.int32 : '-DSRC_TYPE=int',
    np.uint64 : '-DSRC_TYPE=ulong',
    np.int64 : '-DSRC_TYPE=long',
    np.float32 : '-DSRC_TYPE=float'
}

def is_cpu(ctx):
    device = ctx.devices[0]
    return device.get_info(cl.device_info.TYPE) == 2

def DIVUP(a, b):
    return int(math.ceil( float(a) / float(b)))

class VideoPatchSearch():
    """
    Optimized nearest patch search with OpenCL
    """
    def __init__(self, step=1, patch_search_width=7, patch_data_width=7, past_frames=4, future_frames=4, search_width=21, input_dtype=np.float32):
        """
        step: nearest neighbors should be computed every step pixels
        patch_search_width: width of the square patches used to compute the matches
        patch_data_width: width of the square area to fetch at the neighbors positions.
            for example 1 means return only the value of the centers.
        past_frames: How many frames of the past should be used for the search.
        future_frames: How many frames of the future should be used for the search.
        search_width: width of the square region of the center pixels of all compared
            patches on one image. past_frames * future_frames * search_width patches
            are compared. the number of neighbors is 1 + past_frames + future_frames.
        input_dtype: type of the videos on which the search will occur.
        """
        num_neighbors = 1 + past_frames + future_frames
        ctx = cl.create_some_context()
        queue = cl.CommandQueue(ctx)
        f = open('video_patch_search.cl', 'r')
        fstr = "".join(f.readlines())
        build_options = '-cl-mad-enable -cl-unsafe-math-optimizations -cl-fast-relaxed-math -cl-no-signed-zeros'
        build_options += ' -DPATCH_AGGREGATION_STEP='  + ("%d" % step)
        build_options += ' -DPATCH_WIDTH='  + ("%d" % patch_search_width)
        build_options += ' -DNUM_NEIGHBORS=' + ("%d" % num_neighbors)
        build_options += ' -DWINDOW_SEARCH_WIDTH=' + ("%d" % search_width)
        build_options += ' -DWINDOW_SEARCH_FRAMES_PAST=' + ("%d" % past_frames)
        build_options += ' -DWINDOW_SEARCH_FRAMES_FUTURE=' + ("%d" % future_frames)
        build_options += ' ' + src_type_parameter[input_dtype] #video.dtype.type
        if search_width <= 256 and not(is_cpu(ctx)):
            wksize = 128
            build_options += ' -DUSE_CACHE -DWK_SIZE=' + ("%d" %  wksize)
        else:
            wksize = 64
            build_options += ' -DWK_SIZE=64'
        assert(wksize > patch_search_width)
        program = cl.Program(ctx, fstr).build(options=build_options)

        self.compute_nn = program.compute_nearest_neighbors_by_convolution
        self.compute_nn.set_scalar_arg_dtypes([None, None, np.int32, np.int32, np.int32, np.int32, np.int32, np.int32, np.int32, np.int32])

        self.ctx = ctx
        self.queue = queue
        self.step = step
        self.patch_search_width = patch_search_width
        self.patch_data_width = patch_data_width
        self.num_neighbors = num_neighbors
        self.past_frames = past_frames
        self.future_frames = future_frames
        self.input_dtype = input_dtype
        self.wksize = wksize

    def compute(self, video, num_frame):
        """
        input:
           video: numpy array gray image of type input_dtype [num_frames, h, w] or [num_frames, h, w, 1]
           num_frame: frame for which to compute the neighbors
        output:
           numpy array (video height/step, video width/step, num_neighbors) of indices of nearest neighbors.
           [i, j, :] contains the indices of the top left pixel of patches of size patch_data_width * patch_data_width.
           The matches have been computed for patches of size patch_search_width * patch_search_width.
           If [i, j] is the top left pixel of a given search patch, [i, j] points to the top left of the matched
           data patches with same patch center (see comments in the code for more details).
           The indices convert to (y, x) by doing indice / width and indice % width.
        """

        assert(video.dtype.type == self.input_dtype)
        assert(len(video.shape) == 3 or (len(video.shape) == 4 and video.shape[3] == 1))

        nf = video.shape[0]
        h = video.shape[1]
        w = video.shape[2]

        assert(num_frame >= self.past_frames and num_frame < (nf-self.future_frames))
        assert(nf*h*w <= 4294967295) # patch indice overflow (stored on uint32). Use video extract.

        mf = cl.mem_flags
        mp = cl.map_flags

        video_cl = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=video)
        dst_pos_cl = cl.Buffer(self.ctx, mf.WRITE_ONLY, (w // self.step) * (h // self.step) * self.num_neighbors * 4)

        global_size = [h//self.step, DIVUP(w, self.wksize - (self.patch_search_width - 1)) * self.wksize]
        local_size = [1, self.wksize]
        # For naive kernel:
        #global_size = [w, h]
        #local_size = None

        self.compute_nn(self.queue, global_size, local_size, dst_pos_cl, video_cl, w, h, w, w*h, 0, 0, num_frame*w*h, 0)

        dst_pos = np.empty([h // self.step, w // self.step, self.num_neighbors], dtype=np.uint32)
        cl.enqueue_copy(self.queue, dst_pos, dst_pos_cl)

        # At this point, dst_pos[i, j, :] contains the indices (top left of patch_search_width*patch_search_width
        # patches) of matches for the patch with top left (i, j).
        # We now shift such that the [i, j] points to the top left of the patch_data_width*patch_data_width
        # patch centered on the same point than the patch of size patch_search_width*patch_search_width
        # This enables to have
        # anchor = self.patch_width_nn//2
        # nn_patch[y-anchor, x-anchor,:] points to the top left of the patch of size patch_data_width
        # * patch_data_width centered in [y, x].

        if (self.patch_search_width != self.patch_data_width):
            diff_xy = (self.patch_search_width - self.patch_data_width) // 2
            index_offset = diff_xy * w + diff_xy
            dst_pos = dst_pos + index_offset
        return dst_pos

    def build_neighbors_array(self, video, nn):
        """
        input:
           video: numpy array [num_frames, h, w, c] with c the number of channels.
               It doesn't have to be of type input_dtype, but the video size num_frames, h, w should
               match the original, in order to have matching indices.
           nn: extract of indices computed by self.compute()
        output:
           numpy array (nn height, nn width, c * num_neighbors * patch_data_width * patch_data_width)
           containing the neighbors extracted info.
        """
        h = nn.shape[0]
        w = nn.shape[1]
        videoh = video.shape[1]
        videow = video.shape[2]
        c = video.shape[3]

        dst = np.empty([h, w, self.num_neighbors, self.patch_data_width, self.patch_data_width, c], dtype=video.dtype)
        anchor = (self.patch_data_width - 1) // 2
        if (video.dtype == np.uint8):
            build_neighbors_array_accelerated_uint8(dst, video, nn)
        elif (video.dtype == np.float32):
            build_neighbors_array_accelerated_float32(dst, video, nn)
        else:
            for i in range(dst.shape[0]):
                for j in range(dst.shape[1]):
                    indices = nn[i, j, :]
                    for k, ind in enumerate(indices):
                        f = ind // (videow * videoh)
                        x = ind % videow
                        y = (ind-f*videoh*videow) // videow
                        dst[i, j, k, :, :, :] = video[f,y:y+self.patch_data_width,x:x+self.patch_data_width, :]
        dst = dst.reshape([dst.shape[0], dst.shape[1], dst.shape[2]*dst.shape[3]*dst.shape[4]* dst.shape[5]])
        return dst

