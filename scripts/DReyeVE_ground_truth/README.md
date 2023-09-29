## Creating new DReyeVE ground truth

Pre-computed saliency maps can be downloaded [here](https://drive.google.com/drive/folders/1os7jFkPnAm0o5GFAmKC8OOcktIdTZIfI?usp=drive_link).

To reproduce the results, follow the steps outlined in this README.

### Install vlfeat library

Download vlfeat library from <https://github.com/vlfeat/>

Install dependencies 

```
sudo apt-get install intel-mkl-full
```

For vlfeat 0.9.20 follow the advice in https://github.com/vlfeat/vlfeat/issues/214 to compile

For me removing default(none) from vl/kmeans.c worked to some extent.

I also had to comment out the following from toolbox/mexutils.h:
```
/* these attributes suppress undefined symbols warning with GCC */
#ifdef VL_COMPILER_GNUC
#if (! defined(HAVE_OCTAVE))
EXTERN_C void __attribute__((noreturn))
mexErrMsgIdAndTxt (const char * identifier, const char * err_msg, ...) ;
#else
extern void __attribute__((noreturn))
mexErrMsgIdAndTxt (const char *id, const char *s, ...);
#endif
#endif
```

I don't have octave and it was complaining about a missing ; before
```
EXTERN_C void
        ^
```

It then compiled with matlab support on Ubuntu 22.04 with Matlab 2022b

In the vlfeat directory run:

```
make ARCH=glnxa64 MEX=/usr/local/MATLAB/R2022b/bin/mex
```

In MATLAB run

```
vl_setup
```

Run vl_demo

if there is an error `libvl.so: cannot open shared object file: No such file or directory`

add library to path:

```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path_to_vlfeat-0.9.20/toolbox/mex/mexa64/
```

Restart matlab

Run `vl_demo` again

### Project gaze from drivers' view (ETG) to scene view (GAR)

Run matlab script 'project_etg_to_garmin.m' for each video sequence:

```
for seq = 1:74
	tic; project_etg_to_garmin(seq); toc;
end
```

Set `debug` and `show_matches` to True (lines 37-38) to visualize the intermediate results.

This script might take a very long time to run, up to several days, depending on the CPU available. 

Pre-computed homographies are available [here](https://drive.google.com/file/d/19G5RmGajNOJRKaMfZef32EXr6Q6fwCSs/view?usp=drive_link).

### Compute motion-compensated saliency

In the original DR(eye)VE implementation, the fixations are further aggregated over 1s around each frame (so fixations from previous and following 12 frames). For this, a homography was computed between each pair of frames. This is not only very time consuming (over 1 month on a single machine for the entire dataset) but also noisy. 

Here, we follow a different approach of motion-compensated saliency using optical flow. 

#### Compute optical flow maps

Download RAFT algorithm for optical flow from <https://github.com/princeton-vl/RAFT> into the `RAFT` folder.

Update paths to DReyeVE data and RAFT in `RAFT/docker/run_RAFT.sh` script. 

Run `run_RAFT.sh` and inside docker run script `compute_optical_flow.py` for each video in the DReyeVE dataset as follows:

```
	python compute_optical_flow.py --model=models/raft-things.pth --path=DREYEVE_DATA_PATH/07/frames/ --results_path=DREYEVE_DATA_PATH/07/flow_RAFT
```

Note that at 1/2 size (960x540), optical flow maps for each video require approx. 32GB of storage space.


#### Compute new saliency maps

```
python3 create_new_ground_truth.py --seq_num <seq_num> 
```

Options:

```
--seq_num - number of DReyeVE video
--visualize - show the results
--overwrite - overwrite existing results
--scale_factor - resize optical flow
--sigma - size of the Gaussian kernel
```