

from gtsam import Pose3

import gtsfm.utils.io as io_utils



# def test_read_points_txt() -> None:
# 	""" """
# 	fpath = ""
# 	point_cloud, rgb = io_utils.read_points_txt(fpath)

# 	assert isinstance(point_cloud, np.ndarray)
# 	assert isinstance(rgb, np.ndarray)

def test_read_points_txt_nonexistent_file() -> None:
	"""Ensure that providing a path to a nonexistent file returns None for both return args."""
	fpath = "nonexistent_dir/points.txt"
	point_cloud, rgb = io_utils.read_points_txt(fpath)

	assert point_cloud is None
	assert rgb is None


# def test_read_images_txt() -> None:
# 	""" """
# 	fpath = ""
# 	wTi_list, img_fnames = io_utils.read_images_txt(fpath)

# 	assert all([isinstance(wTi, Pose3) for wTi in wTi_list])
# 	assert all([isinstance(img_fname, str) for img_fname in img_fnames])


def test_read_images_txt_nonexistent_file() -> None:
	"""Ensure that providing a path to a nonexistent file returns None for both return args."""
	fpath = "nonexistent_dir/images.txt"
	wTi_list, img_fnames = io_utils.read_images_txt(fpath)
	assert wTi_list is None
	assert img_fnames is None

# def test_read_cameras_txt() -> None:
# 	""" """
# 	calibrations = io_utils.read_cameras_txt(fpath: str) -> Optional[List[Cal3Bundler]]:


def test_read_cameras_txt_nonexistent_file() -> None:
	""" """
	fpath = "nonexistent_dir/cameras.txt"
	calibrations = io_utils.read_cameras_txt(fpath)
	assert calibrations is None
	
