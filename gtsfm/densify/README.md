Evaluation of sparse reconstruction results

- `mvs_base.py` base class for MVS methods
- `/thirdparty/patchmatchnet` a trunk of Wang et al.'s work, [PatchmatchNet](https://github.com/FangjinhuaWang/PatchmatchNet.git).
  - `datasets` dataset utils for PatchMatch Net
  - `models` detailed model architecture for PatchMatch Net
    - `module.py` basic modules like convolution blocks
    - `patchmatch.py` patchmatch structure, including Initialization and Local Perturbation,
        Adaptive Propagation, Adaptive Evaluation, etc.
    - `net.py` assemble all modules of PatchMatch Net
  - `utils.py` PatchMatch Net utils
