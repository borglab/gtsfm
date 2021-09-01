# PatchmatchNet (CVPR2021 Oral)
official source code of paper 'PatchmatchNet: Learned Multi-View Patchmatch Stereo'


## Introduction
PatchmatchNet is a novel cascade formulation of learning-based Patchmatch which aims at decreasing memory consumption and computation time for high-resolution multi-view stereo. If you find this project useful for your research, please cite: 

```
@misc{wang2020patchmatchnet,
      title={PatchmatchNet: Learned Multi-View Patchmatch Stereo}, 
      author={Fangjinhua Wang and Silvano Galliani and Christoph Vogel and Pablo Speciale and Marc Pollefeys},
      journal={CVPR},
      year={2021}
}
```

## Installation

### Note:
`--patchmatch_iteration` represents the number of iterations of Patchmatch on multi-stages (e.g., the default number `1,2,2` means 1 iteration on stage 1, 2 iterations on stage 2 and 2 iterations on stage 3). `--propagate_neighbors` represents the number of neighbors for adaptive propagation (e.g., the default number `0,8,16` means no propagation for Patchmatch on stage 1, using 8 neighbors for propagation on stage 2 and using 16 neighbors for propagation on stage 3). As explained in our paper, we do not include adaptive propagation for the last iteration of Patchmatch on stage 1 due to the requirement of photometric consistency filtering. So in our default case (also for our pretrained model), we set the number of propagation neighbors on stage 1 as `0` since the number of iteration on stage 1 is `1`. If you want to train the model with more iterations on stage 1, change the corresponding number in `--propagate_neighbors` to include adaptive propagation for Patchmatch expect for the last iteration.

## Acknowledgements
This project is done in collaboration with "Microsoft Mixed Reality & AI Zurich Lab". 
Thanks to Yao Yao for opening source of his excellent work [MVSNet](https://github.com/YoYo000/MVSNet). Thanks to Xiaoyang Guo for opening source of his PyTorch implementation of MVSNet [MVSNet-pytorch](https://github.com/xy-guo/MVSNet_pytorch).
