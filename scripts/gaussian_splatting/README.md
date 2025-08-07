# Initial Implementation of Gaussian Splatting

## Initial Steps
Before being able to run the Gaussian Splatting, you need to perform the following steps (details for all these steps are in the main README):
* Clone the GTSfM repository
* Create a conda environment with the required dependencies
* Install GTSfM as a module
* Download the model weights
* Get the SfM points

## Training

You can choose to start from random points by skipping the `init_type` argument. Other arguments can be altered similarly. By default, every sixth image will be part of the validation set; you can change it by adding `--test_every -1` to your command.
```
CUDA_VISIBLE_DEVICES=0 python scripts/gaussian_splatting/custom_trainer.py default --data_dir results/ba_output --images_dir tests/data/set1_lund_door/images/ --init_type sfm --max_steps 10000
```

### These scripts are not maintained and are just meant for the purposes of initial experimentation or understanding.