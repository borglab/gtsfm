"""Utilities for GtsfmData preprocessing and manipulation.

Authors: GitHub Copilot
"""

from gtsfm.common.gtsfm_data import GtsfmData
from gtsfm.utils import logger as logger_utils

logger = logger_utils.get_logger()


def remove_cameras_with_no_tracks(scene: GtsfmData, context: str = "BA") -> tuple[GtsfmData, bool]:
    """Remove cameras with no tracks from a scene.

    Args:
        scene: The scene to remove cameras from.
        context: Context string for logging (e.g., "parent BA", "node-level BA"). Defaults to "BA".

    Returns:
        A tuple containing the scene with cameras removed and a boolean indicating if the scene should run BA.
    """
    all_cameras = set(scene.get_valid_camera_indices())
    camera_measurement_map = scene.get_camera_to_measurement_map()
    cameras_with_measurements = set(camera_measurement_map.keys())
    zero_track_cameras = sorted(all_cameras - cameras_with_measurements)
    
    if zero_track_cameras:
        logger.warning("📋 Cameras with zero tracks before %s: %s", context, zero_track_cameras)
        if cameras_with_measurements:
            scene = GtsfmData.from_selected_cameras(scene, sorted(cameras_with_measurements), keep_all_image_infos=True)
            logger.info(
                "Pruned %d zero-track cameras; %d cameras remain for %s.",
                len(zero_track_cameras),
                len(scene.get_valid_camera_indices()),
                context,
            )
        else:
            logger.warning("All cameras lack tracks; skipping %s.", context)
            return scene, False
    else:
        logger.info("✅ All cameras have at least one track before %s.", context)

    return scene, True
