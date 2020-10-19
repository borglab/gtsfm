
import gtsam
from gtsam import readBal
import numpy as np

from bundle.bundle_adjustment import BundleAdjustmentBase

def test_ba_dubrovnik():
	""" """
	input_file_name = "dubrovnik-3-7-pre"
	input_file = gtsam.findExampleDataFile(input_file_name)

	# Load the SfM data from file
	scene_data = readBal(input_file)

	ba_obj = BundleAdjustmentBase()
	error = ba_obj.run(scene_data)
	expected_error = 0.046137573704557046
	assert np.isclose(expected_error, error)


if __name__ == '__main__':
	test_ba_dubrovnik()

