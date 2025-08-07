# Gaussian Splatting integrating into the GTSfM pipeline

Currently only works for linux environments because the rasterization and densification strategies imported from ```gsplat``` are implemented using NVIDIA CUDA.

## How to run
You need to pass ```--run_gs``` flag in your command to run the Gaussian Splatting code.

Currently the code defaults to the ```gtsfm/configs/gaussian_splatting/base_gs.yaml``` to load the arguments for Gaussian Splatting. In order to load arguments from your personal yaml file you will need to add the file to the the ```gtsfm/configs/gaussian_splatting/``` folder and then pass the filename in the command with the flag ```--gaussian_splatting_config_name```.