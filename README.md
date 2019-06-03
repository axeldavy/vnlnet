# vnlnet
VNLnet is a Video denoising CNN with Non-locality information

More details in our paper:
Non-Local Video Denoising by CNN

https://arxiv.org/abs/1811.12758

If you use any part of our code for another project, please cite our paper.


The code to reproduce the training database is available here:
https://github.com/cmla/f16-video-set

The code to reproduce our davis test-dev test set is available here:
https://github.com/cmla/downscaled-davis-dataset

The data of the second testing set should be available soon.


Similarly, if you use any part of these codes for another project, please cite our paper.


**** Requirements ****


pyTorch >= 0.4.0
numpy
imageio
tifffile
pyopencl

OpenCL >= 1.2
CUDA (for training)

You can install the python requirements with:
pip install -r requirements.txt --user

**** General Info ****


The patch search is implemented in OpenCL. To run the code, a valid OpenCL driver is required. A CPU OpenCL driver can be used for testing, even though the code was tuned for GPUs.
For example, on Ubuntu and NVidia GPU, you must install nvidia-opencl-dev.

If using CUDA, please notice that CUDA_VISIBLE_DEVICES also affects the visible OpenCL devices.


**** Training Info ****


Assuming the training and validation data are in directories train/ and val/, the training command is:
python3 train.py --epochs 20 --milestone 12 17 --sigma 20 --oracle_mode 0 --past_frames 7 --future_frames 7 --search_window_width 41 --nn_patch_width 41 --pass_nn_value --save_dir vnlnet_gray_20 --train_dir train/ --val_dir val/

The parameters are described in train.py. More particularly, oracle_mode describes whether to use the noise free image for the patch search (but not for the pixel data). This enables to evaluate the performance gap compared to if a perfect patch matching was available.

You might want to previously set PYOPENCL_CTX (for example export PYOPENCL_CTX=0) to define the OpenCL device to use for the patch search code.


Pretrained models:

This repository contains several pretrained models.
For example, vnlnet_gray_20 corresponds to vnlnet trained on gray images for noise 20.
The same naming pattern is used for the other files.


**** Testing Info ****


The video sequence to denoise must be extracted in a directory.
Assuming the video to denoise is alone in the directory inputs/, you can denoise the sequence with:

python3 test.py directory --net vnlnet_gray_20.pth --input inputs/ --output outputs/

Assuming vnlnet_gray_20 corresponds to best network for your estimated noise.

For benchmarking purposes, a noise-free sequence can be given with the parameter --add-noise to
add a noise of standard deviation corresponding to what the network was trained for.
If passing sequences with already generated synthetic noise for comparison with another method,
it is more accurate to pass tiff files encoding floating point data.

If only the result on one image matters, the argument --only_frame can be used to specify the only
frame to denoise.

If testing on a system without CUDA, --cpu can be used to run the network on cpu. It still requires
a working OpenCL driver (which can be CPU or GPU, but we recommand a powerful GPU).

Assuming the video is not alone in the directory inputs (and is named with the pattern video_001.png, ... video_300.png),
or only a subset of frames need to be denoised (for example from 100 to 199),
this command line can be used:

python3 test.py pattern --net vnlnet_gray_20.pth --input_pattern inputs/video_%03d.png --output outputs/video_%03d.png --first 100 --last 199


**** Scoring Info ****


To compute the PSNR or the SSIM on some video sequence, we included several scripts.

psnr.py computes the psnr for a video sequence
ssim.py computes the ssim score for a video sequence
