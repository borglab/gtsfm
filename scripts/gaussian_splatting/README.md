# Initial Implementation of Gaussian Splatting

## Initial Steps
Before being able to run the Gaussian Splatting, you need to perform the following steps (details for all these steps are in the main README):
* Clone the GTSfM repository
* Create a conda environment with the required dependencies
* Install GTSfM as a module
* Download the model weights
* Get the SfM points
* Convert the camera poses estimated by GTSfM to [nerfstudio](https://docs.nerf.studio/en/latest/) format:

## Additional Dependencies

* Check your torch version with the CUDA version on your system and set the appropriate paths.
* Install the additional necessary dependencies
```
pip install git+https://github.com/nerfstudio-project/gsplat.git
pip install git+https://github.com/rahul-goel/fused-ssim/ --no-build-isolation
pip install torchmetrics==1.7.3
pip install tyro==0.9.24
```

## Training

You can choose to start from random points by skipping the `init_type` argument. Other arguments can be altered similarly. By default, every sixth image will be part of the validation set; you can change it by adding `--test_every 10000` to your command. (basically an integer more than the number of images) 
```
CUDA_VISIBLE_DEVICES=0 python scripts/gaussian_splatting/custom_trainer.py default --data_dir results/nerfstudio_input --init_type sfm --max_steps 5000
```

## Rendering

In order to test the rendering function, a test `custom_traj_path` file is provided. You can render right after training by changing the training command by adding the ` --custom_traj_path scripts/gaussian_splatting/my_new_custom_path.json` or you can run the following command.

```
 CUDA_VISIBLE_DEVICES=0 python scripts/gaussian_splatting/custom_trainer.py default --data_dir results/nerfstudio_input --custom_traj_path scripts/gaussian_splatting/my_new_custom_path.json --ckpt results/custom_dataset/ckpts/ckpt_10000.pt
```