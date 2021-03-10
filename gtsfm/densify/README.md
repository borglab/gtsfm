# Densify Module of GTSFM

#### Usage

* At gtsfm root directory, `python gtsfm/runner/run_scene_optimizer_densify.py`
* API
  * ```python 
    from gtsfm.densify.mvsnets.mvsnets import MVSNets

    MVSNets.densify(sfm_result.sfm_data, 
                    image_path=os.path.join(DATA_ROOT, "set1_lund_door"), 
                    image_extension="JPG",
                    view_number=5,
                    thres=[1.0, 0.01, 0.8])
    ```
* output path `results_densify/outputs/`
  * `results_densify/outputs/scan1/confidence`: .pfm confidence map from each view
  * `results_densify/outputs/scan1/depth_est`: .pfm expected depth map from each view
  * `results_densify/outputs/scan1/depth_img`: heatmap (.png) files of (.pfm) files above
  * `results_densify/outputs/scan1/mask`: masks of photo/geo/final used for mesh construction
  * `results_densify/*.ply`: generated mesh

#### Sample output

<div sytle="text-align:center;">
<figure style="display: inline-block; text-align: center;">
  <img src="docs/img/gt_depth_04.png" width="200" style="margin:0px;"/>
  <figcaption>Depthmap use accurate cameras</figcaption>
</figure>
<figure style="display: inline-block; text-align: center;">
  <img src="docs/img/gt_depth_04.png" width="200" style="margin:0px;"/>
  <figcaption>Depthmap use calculated cameras</figcaption>
</figure>
<figure style="display: inline-block; text-align: center;">
  <img src="docs/img/res-gt-cam.png" width="200" style="margin:0px;"/>
  <figcaption>Mesh use accurate cameras</figcaption>
</figure>
<figure style="display: inline-block; text-align: center;">
  <img src="docs/img/res-gen-cam.png" width="200" style="margin:0px;"/>
  <figcaption>Mesh use calculated cameras</figcaption>
</figure>
</div>



#### TODOs & Graphs

- [ ] Add unit tests
- [ ] Find metrics
- [ ] Mesh Refinement

![TODOs](docs/img/den1.png)

![Graph](docs/img/den2.png)

#### References

* Currently the densify module is based on the work of Fangjinhua Wang et al., PatchmatchNet: [Learned Multi-View Patchmatch Stereo](https://arxiv.org/abs/2012.01411). Codes in `densify/mvsnets/source/PatchmatchNet` is mainly based on its corresponding [repository](https://github.com/FangjinhuaWang/PatchmatchNet).

* The View Selection part (in `densify/mvsnets/mvsUtils.py`) is based on Yao et al.'s work [MVSNet: Depth Inference for Unstructured Multi-view Stereo](https://arxiv.org/abs/1804.02505).

