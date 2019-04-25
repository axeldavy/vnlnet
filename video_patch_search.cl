/**************************************************************************
 *
 * Copyright 2018 Axel Davy <axel.davy@ens-cachan.fr>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.

 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.

 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 **************************************************************************/

#define START_WINDOW ((int)(-WINDOW_SEARCH_WIDTH + (WINDOW_SEARCH_WIDTH+1)/2))
#define END_WINDOW (WINDOW_SEARCH_WIDTH + START_WINDOW)

/* No patch can reach this weight */
#define UNUSED_WEIGHT ((float)(-1.f))

#define PATCH_SIZE (PATCH_WIDTH*PATCH_WIDTH)


/* To enable float16. Can be faster or slower depending on configuration. */
#if defined(USE_FLOAT16) && defined(cl_khr_fp16)
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#define WT half
#define WT4 half4
#define WT8 half8
#define convert_WT convert_half
#define convert_WT4 convert_half4
#define convert_WT8 convert_half8
#ifndef HALF_MAX
#define HALF_MAX 0x1.ffcp15h
#endif
#define MAXVAL HALF_MAX
#define SIZEOF_WT 2
#else
#define WT float
#define WT4 float4
#define WT8 float8
#define convert_WT convert_float
#define convert_WT4 convert_float4
#define convert_WT8 convert_float8
#define MAXVAL MAXFLOAT
#define SIZEOF_WT 4
#endif

#define SIZEOF_INT 4

/* Disable because some compilers
 * pass __OPENCL_C_VERSION__ >= 200,
 * but do not support the attribute */
#if 0 && __OPENCL_C_VERSION__ >= 200
#define barrier(x) work_group_barrier(x)
#define unroll_loop() __attribute__((opencl_unroll_hint))
/* Defined by C99 - which Opencl C follows */
#elif defined(_Pragma)
#define unroll_loop() _Pragma("unroll")
/* In case this variant is used */
#elif defined(__Pragma)
#define unroll_loop() __Pragma("unroll")
/* Bad compiler */
#else
#define unroll_loop()
#endif

#define DIVUP(x,y) (((x)+(y)-1)/(y))

#define MAX2(x,y) ((x)>(y) ? (x) : (y))

#define sum(x,y) ((x)+(y))

#define reduce_op_on_table(res, op, name, length_table) \
    res = name[0]; \
    unroll_loop() \
    for (int it = 1; it < length_table; it++) { \
        res = op(res, name[it]); \
    }


#define maintain_best_weight(best_nn_patches_weight, \
                              best_nn_patches_offset, \
                              should_check_overlap, \
                              patch_weight, \
                              patch_offset) \
unroll_loop() \
for (int mbw_for = 0; mbw_for < 1; mbw_for++) { \
    if (should_check_overlap) { \
        if (best_nn_patches_offset == patch_offset) { \
            break; \
        } \
    } \
 \
    if (patch_weight < best_nn_patches_weight) { \
        best_nn_patches_weight = patch_weight; \
        best_nn_patches_offset = patch_offset; \
    } \
}

#define maintain_best_weights(best_nn_patches_weights, \
                              best_nn_patches_offsets, \
                              num_neighbors_to_track, \
                              should_check_overlap, \
                              should_check_threshold, \
                              threshold_search, \
                              patch_weight, \
                              patch_offset) \
unroll_loop() \
for (int mbw_for = 0; mbw_for < 1; mbw_for++) { \
    /* These two ifs should be removed at compilation, because the \
     * booleans values are always the same. Similarly, unrolling \
     * is possible because num_neighbors_to_track is fixed. */ \
    if (should_check_overlap) { \
        bool mbw_already_there = false; \
        unroll_loop() \
        for (int mbw_i = 0; mbw_i < num_neighbors_to_track; mbw_i++) { \
            mbw_already_there = (best_nn_patches_offsets[mbw_i] == patch_offset) ? true : mbw_already_there; \
        } \
        if (mbw_already_there) { \
            break; \
        } \
    } \
 \
    if (should_check_threshold) { \
        if (patch_weight > convert_WT(threshold_search)) { \
            break; \
        } \
    } \
 \
    /* Maintain ordered best weights. Note they are positive. */ \
    if (patch_weight < best_nn_patches_weights[num_neighbors_to_track-1]) { \
        unroll_loop() \
        for (int mbw_i = num_neighbors_to_track-1; mbw_i >= 0; mbw_i--) { \
            if (mbw_i > 0) { \
                bool cond = (best_nn_patches_weights[mbw_i-1] <= patch_weight); \
                best_nn_patches_weights[mbw_i] = select(best_nn_patches_weights[mbw_i-1], patch_weight, cond); \
                best_nn_patches_offsets[mbw_i] = select(best_nn_patches_offsets[mbw_i-1], patch_offset, cond); \
                if (cond) break;\
            } else { \
                best_nn_patches_weights[mbw_i] = patch_weight; \
                best_nn_patches_offsets[mbw_i] = patch_offset; \
            } \
        } \
    } \
}


inline void write_neighbor_index(__global int * restrict dst_pos,
                            int offset,
                            int num_neighbor,
                            int dst_offset,
                            int w,
                            int y_corner_top_left,
                            int x_corner_top_left)
{
    dst_pos[dst_offset+((w / PATCH_AGGREGATION_STEP)*(y_corner_top_left / PATCH_AGGREGATION_STEP)+(x_corner_top_left / PATCH_AGGREGATION_STEP))*NUM_NEIGHBORS+num_neighbor] = offset;
}




#ifndef WK_SIZE
#define WK_SIZE 64
#endif

/* Number of times WK_SIZE is needed to store WK_SIZE + WINDOW_SEARCH_WIDTH
 * assumes WK_SIZE >= WINDOW_SEARCH_WIDTH */
#define PATCH_CACHE_ROW_WIDTH_DIV_WK_SIZE DIVUP(WK_SIZE + WINDOW_SEARCH_WIDTH, WK_SIZE)
#define PATCH_CACHE_ROW_WIDTH (WK_SIZE * PATCH_CACHE_ROW_WIDTH_DIV_WK_SIZE)

#if USE_CACHE && WK_SIZE < WINDOW_SEARCH_WIDTH
#error invalid assumptions for kernel
#endif


__kernel __attribute__((reqd_work_group_size(1, WK_SIZE, 1))) void compute_nearest_neighbors_by_convolution(__global int * restrict dst_pos,
                                                                                                             __global const SRC_TYPE * restrict src,
                                                                                                             int w,
                                                                                                             int h,
                                                                                                             int items_row,
                                                                                                             int items_img,
                                                                                                             int search_offset_x,
                                                                                                             int search_offset_y,
                                                                                                             int src_offset,
                                                                                                             int dst_offset)
{
#ifdef USE_CACHE
    __local SRC_TYPE image_cache[PATCH_CACHE_ROW_WIDTH*PATCH_WIDTH];
#endif
    __local WT weight_share[WK_SIZE+PATCH_WIDTH];
    int id = get_local_id(1);
    bool not_overlap = (id >= 0 && id <= WK_SIZE - PATCH_WIDTH);
    int x_corner_top_left = id + (WK_SIZE - (PATCH_WIDTH - 1)) * get_group_id(1) + search_offset_x;
    int y_corner_top_left = PATCH_AGGREGATION_STEP*get_global_id(0) + search_offset_y;
    bool is_patch_first_column_inside_image = (x_corner_top_left < w) && (y_corner_top_left + PATCH_WIDTH <= h);
    bool is_patch_inside_image = (x_corner_top_left <= w-PATCH_WIDTH) && (y_corner_top_left + PATCH_WIDTH <= h);
    bool write_result = (x_corner_top_left % PATCH_AGGREGATION_STEP == 0) && is_patch_inside_image && not_overlap;

    /* We are allowed to do that even if we use barriers,
     * because all work items in the work group
     * have same y_corner_top_left, and thus
     * we maintain that all items in a group execute the same
     * barriers. */
    if (y_corner_top_left + PATCH_WIDTH > h) {
        return;
    }

    __private WT main_patch_center_column[PATCH_WIDTH];
    #pragma unroll
    for (int row = 0; row < PATCH_WIDTH; row++) {
        main_patch_center_column[row] = convert_WT(src[src_offset + clamp(items_row*(y_corner_top_left+row)+x_corner_top_left, 0, items_img-1)]);
    }

    for (int dz = -WINDOW_SEARCH_FRAMES_PAST; dz <= WINDOW_SEARCH_FRAMES_FUTURE; dz++) {
        __global const SRC_TYPE * restrict src_comp = src + src_offset + dz * items_img;
        int best_nn_patches_offset = 0x7FFFFFFF;
        WT best_nn_patches_weight = INFINITY;
        if (dz == 0) {
            best_nn_patches_offset = src_offset+items_row*(y_corner_top_left)+x_corner_top_left;
            if (write_result) {
                write_neighbor_index(dst_pos,
                               best_nn_patches_offset,
                               dz+WINDOW_SEARCH_FRAMES_PAST,
                               dst_offset,
                               w, y_corner_top_left, x_corner_top_left);
            }
            continue;
        }
#ifdef USE_CACHE
        for (int row = 0; row < PATCH_WIDTH; row++) {
                #pragma unroll
                for (int i = 0; i < PATCH_CACHE_ROW_WIDTH_DIV_WK_SIZE; i++) {
                    image_cache[row*PATCH_CACHE_ROW_WIDTH + id + i*WK_SIZE] = src_comp[clamp(items_row*(y_corner_top_left+row+START_WINDOW-1)+x_corner_top_left+START_WINDOW+i*WK_SIZE, 0, items_img-1)];
                }
            }
        int first_row_index = 0;
#endif
        for (int dy = START_WINDOW; dy < END_WINDOW; dy++) {
#ifdef USE_CACHE
            #pragma unroll
            for (int i = 0; i < PATCH_CACHE_ROW_WIDTH_DIV_WK_SIZE; i++) {
                image_cache[first_row_index*PATCH_CACHE_ROW_WIDTH + id + i*WK_SIZE] = src_comp[clamp(items_row*(y_corner_top_left+dy+PATCH_WIDTH-1)+x_corner_top_left+START_WINDOW+i*WK_SIZE, 0, items_img-1)];
            }
            first_row_index = (first_row_index+1)%PATCH_WIDTH;
            barrier(CLK_LOCAL_MEM_FENCE);
#endif
            for (int dx = START_WINDOW; dx < END_WINDOW; dx++) {
                bool is_compared_patch_first_column_inside_image = (x_corner_top_left+dx >= 0) && (y_corner_top_left+dy >= 0) && (x_corner_top_left+dx < w) && (y_corner_top_left+dy + PATCH_WIDTH <= h);
                WT patch_weight = 0.f;
                #pragma unroll
                for (int row = 0; row < PATCH_WIDTH; row++) {
#ifdef USE_CACHE
                    WT compared_data = convert_WT(image_cache[((row+first_row_index)%PATCH_WIDTH)*PATCH_CACHE_ROW_WIDTH + id + (dx-START_WINDOW)]);
#else
                    WT compared_data = convert_WT(src_comp[clamp(items_row*(y_corner_top_left+dy+row)+x_corner_top_left+dx, 0, items_img-1)]);
#endif
#ifdef NORM_L1
                    patch_weight += fabs(compared_data - main_patch_center_column[row]);
#else
                    WT diff = compared_data - main_patch_center_column[row];
                    patch_weight += diff * diff;
#endif
                }
                patch_weight = (is_patch_first_column_inside_image && is_compared_patch_first_column_inside_image) ? patch_weight : INFINITY;
                weight_share[id] = patch_weight;
                barrier(CLK_LOCAL_MEM_FENCE);
                #pragma unroll
                for (int col = 1; col < PATCH_WIDTH; col++) {
                    patch_weight += weight_share[id+col];
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                if (write_result) {
                    maintain_best_weight(best_nn_patches_weight,
                                         best_nn_patches_offset,
                                         false,
                                         patch_weight,
                                         src_offset+items_img*dz+items_row*(y_corner_top_left+dy)+x_corner_top_left+dx);
                }
            }
        }
        if (write_result) {
            write_neighbor_index(dst_pos,
                           best_nn_patches_offset,
                           dz+WINDOW_SEARCH_FRAMES_PAST,
                           dst_offset,
                           w, y_corner_top_left, x_corner_top_left);
        }
    }
}
