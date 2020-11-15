
import gtsam
from gtsam import readBal
import numpy as np

import dask

from bundle.bundle_adjustment import BundleAdjustmentBase

def test_ba_dubrovnik():
    """ """
    input_file_name = "dubrovnik-3-7-pre"
    input_file = gtsam.findExampleDataFile(input_file_name)

    # Load the SfM data from file
    scene_data = readBal(input_file)

    ba_obj = BundleAdjustmentBase()
    computed_error = ba_obj.create_computation_graph(scene_data)
	
    with dask.config.set(scheduler='single-threaded'):
        dask_error = dask.compute(computed_error)[0]

    expected_error = 0.046137573704557046
    assert np.isclose(expected_error, dask_error)


if __name__ == '__main__':
    test_ba_dubrovnik()

