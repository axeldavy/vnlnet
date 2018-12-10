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

#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: nonecheck=False
#cython: embedsignature=False
#cython: cdivision=True
#cython: cdivision_warnings=False
#cython: always_allow_keywords=False
#cython: profile=False
#cython: infer_types=False

cimport cython
import numpy as np
cimport numpy as np
from libc.string cimport memcpy

def build_neighbors_array_accelerated_float32(np.ndarray[np.float32_t, ndim=6] dst, np.ndarray[np.float32_t, ndim=4] video, np.ndarray[np.uint32_t, ndim=3] nn):
        cdef int nf = video.shape[0]
        cdef int h = video.shape[1]
        cdef int w = video.shape[2]
        cdef int c = video.shape[3]
        cdef int patch_width = dst.shape[4]
        cdef int num_neighbors = dst.shape[2]
        cdef int i, j, x, y, f, k, ind
        cdef int u, v, t
        cdef int l1 = dst.shape[0]
        cdef int l2 = dst.shape[1]

        cdef int num_pix_image = w * h

        if dst[0, 0, 0, 0, :, :].flags.c_contiguous == True and video[0, 0, :, :].flags.c_contiguous == True:
            for i in range(l1):
                for j in range(l2):
                    for k in range(num_neighbors):
                        ind = nn[i, j, k]
                        f = ind // num_pix_image
                        ind -= f * num_pix_image;
                        y = ind // w
                        ind -= y * w
                        x = ind
                        for u in range(patch_width):
                            memcpy(&dst[i, j, k, u, 0, 0], &video[f, y+u, x, 0], patch_width*c*4)
        else:
            for i in range(l1):
                for j in range(l2):
                    for k in range(num_neighbors):
                        ind = nn[i, j, k]
                        f = ind // num_pix_image
                        ind -= f * num_pix_image;
                        y = ind // w
                        ind -= y * w
                        x = ind
                        for u in range(patch_width):
                            for v in range(patch_width):
                                for t in range(c):
                                    dst[i, j, k, u, v, t] = video[f, y+u, x+v, t]

def build_neighbors_array_accelerated_uint8(np.ndarray[np.uint8_t, ndim=6] dst, np.ndarray[np.uint8_t, ndim=4] video, np.ndarray[np.uint32_t, ndim=3] nn):
        cdef int nf = video.shape[0]
        cdef int h = video.shape[1]
        cdef int w = video.shape[2]
        cdef int c = video.shape[3]
        cdef int patch_width = dst.shape[4]
        cdef int num_neighbors = dst.shape[2]
        cdef int i, j, x, y, f, k, ind
        cdef int u, v, t
        cdef int l1 = dst.shape[0]
        cdef int l2 = dst.shape[1]

        cdef int num_pix_image = w * h

        if dst[0, 0, 0, 0, :, :].flags.c_contiguous == True and video[0, 0, :, :].flags.c_contiguous == True:
            for i in range(l1):
                for j in range(l2):
                    for k in range(num_neighbors):
                        ind = nn[i, j, k]
                        f = ind // num_pix_image
                        ind -= f * num_pix_image;
                        y = ind // w
                        ind -= y * w
                        x = ind
                        for u in range(patch_width):
                            memcpy(&dst[i, j, k, u, 0, 0], &video[f, y+u, x, 0], patch_width*c)
        else:
            for i in range(l1):
                for j in range(l2):
                    for k in range(num_neighbors):
                        ind = nn[i, j, k]
                        f = ind // num_pix_image
                        ind -= f * num_pix_image;
                        y = ind // w
                        ind -= y * w
                        x = ind
                        for u in range(patch_width):
                            for v in range(patch_width):
                                for t in range(c):
                                    dst[i, j, k, u, v, t] = video[f, y+u, x+v, t]
