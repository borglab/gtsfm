"""Dummy translation averaging which just echoes absolute prior values back.

Authors: Ayush Baid
"""
# import abc
# from typing import Dict, List, Optional, Tuple

# import dask
# from dask.delayed import Delayed
# from gtsam import Point3, Pose3, Rot3, Unit3

# from gtsfm.averaging.translation.translation_averaging_base import TranslationAveragingBase
# from gtsfm.common.pose_prior import PosePrior
# from gtsfm.evaluation.metrics import GtsfmMetric, GtsfmMetricsGroup


# class DummyTranslationAveraging(TranslationAveragingBase):
#     def run(
#         self,
#         num_images: int,
#         i2Ui1_dict: Dict[Tuple[int, int], Optional[Unit3]],
#         wRi_list: List[Optional[Rot3]],
#         relative_pose_priors: Dict[Tuple[int, int], PosePrior],
#         wTi_initial: List[Optional[PosePrior]],
#         scale_factor: float = 1,
#         gt_wTi_list: Optional[List[Optional[Pose3]]] = None,
#     ) -> Tuple[List[Optional[Point3]], Optional[GtsfmMetricsGroup]]:
#         wTi_results = [wTi_prior.value.translation() if wTi_prior is not None else None for wTi_prior in wTi_initial]

#         ta_metrics = [
#             GtsfmMetric("num_translations_estimated", len([wti for wti in wTi_results if wti is not None])),
#         ]

#         return wTi_results, GtsfmMetricsGroup("translation_averaging_metrics", ta_metrics)
