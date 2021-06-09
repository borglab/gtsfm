


import itertools

import matplotlib.pyplot as plt
import numpy as np
from argoverse.utils.json_utils import read_json_file
from scipy.spatial.transform import Rotation

import gtsfm.utils.geometry_comparisons as comp_utils
import gtsfm.utils.logger as logger_utils

from gtsam import Rot3, Pose3, Unit3

logger = logger_utils.get_logger()


# find all triplets

"""
See: TheiaSfM/src/theia/math/graph/triplet_extractor.h

https://github.com/sweeneychris/TheiaSfM/blob/master/src/theia/sfm/filter_view_graph_cycles_by_rotation.cc
"""

CYCLE_ERROR_THRESHOLD = 5.0

def filter_to_cycle_consistent_edges(i2Ri1_dict, i2Ui1_dict, two_view_reports_dict, visualize: bool = False):
	""""
	Will return only a subset of these two dictionaries

	"""
	# they should have the same keys
	edges = []
	for (i1,i2), i2Ri1 in i2Ri1_dict.items():
		edge = {
			"i1": i1,
			"i2": i2,
			"i2Ri1": i2Ri1,
			'rotation_angular_error': two_view_reports_dict[(i1,i2)].R_error_deg,
			'translation_angular_error': two_view_reports_dict[(i1,i2)].U_error_deg
		}
		edges.append(edge)

	# triplets = []

	# triplets = itertools.product( range(num_images), range(num_images), range(num_images) )
	# triplets = list(triplets)

	# import pdb; pdb.set_trace()
	# # for edge_info in data:

	# check the cumulative translation/rotation errors between triplets to throw away cameras
	cycle_errors = []
	average_rot_errors = []
	average_trans_errors = []

	cycles_seen = set()

	# (i1,i2) pairs
	cycle_consistent_keys = set()

	for edge_i_info in edges:

		if not edge_i_info["i2Ri1"]:
			continue
		edge_i_keys = [edge_i_info["i1"], edge_i_info["i2"]]

		for edge_j_info in edges:

			if not edge_j_info["i2Ri1"]:
				continue
			edge_j_keys = [edge_j_info["i1"], edge_j_info["i2"]]

			for edge_k_info in edges:

				if not edge_k_info["i2Ri1"]:
					continue
				edge_k_keys = [edge_k_info["i1"], edge_k_info["i2"]]

				cycle_nodes = set(edge_i_keys).union(set(edge_j_keys)).union(set(edge_k_keys))

				unique_edges = set([ tuple(edge_i_keys), tuple(edge_j_keys), tuple(edge_k_keys) ])
				edges_are_unique = len(unique_edges) == 3


				if len(cycle_nodes) == 3 and edges_are_unique:
					
					#import pdb; pdb.set_trace()
					if tuple(sorted(cycle_nodes)) in cycles_seen:
						continue
					else:
						cycles_seen.add(tuple(sorted(cycle_nodes)))

					cycle_error, average_rot_error, average_trans_error = compute_cycle_error(i2Ri1_dict, cycle_nodes, edge_i_info, edge_j_info, edge_k_info)
					
					if cycle_error < CYCLE_ERROR_THRESHOLD:

						cycle_consistent_keys.add(tuple(edge_i_keys))
						cycle_consistent_keys.add(tuple(edge_j_keys))
						cycle_consistent_keys.add(tuple(edge_k_keys))


					cycle_errors.append(cycle_error)
					average_rot_errors.append(average_rot_error)
					average_trans_errors.append(average_trans_error)

				else:
					#print("Not a cycle: Nodes", cycle_edges, " Edges: ", edge_i_keys, edge_j_keys, edge_k_keys)
					pass

	if visualize:
		plt.scatter(cycle_errors, average_rot_errors)
		plt.xlabel("Cycle error")
		plt.ylabel("Avg. Rot3 error over cycle triplet")
		plt.show()

		plt.scatter(cycle_errors, average_trans_errors)
		plt.xlabel("Cycle error")
		plt.ylabel("Avg. Unit3 error over cycle triplet")
		plt.show()


	print("cycle_consistent_keys", cycle_consistent_keys)
	i2Ri1_dict_consistent, i2Ui1_dict_consistent = {}, {}
	for (i1,i2) in cycle_consistent_keys:

		if two_view_reports_dict[(i1,i2)].R_error_deg > 3 or two_view_reports_dict[(i1,i2)].U_error_deg > 3:
			continue

		i2Ri1_dict_consistent[(i1,i2)] = i2Ri1_dict[(i1,i2)]
		i2Ui1_dict_consistent[(i1,i2)] = i2Ui1_dict[(i1,i2)]

	num_consistent_rotations = len(i2Ri1_dict_consistent)
	logger.info("Found %d consistent rotations", num_consistent_rotations)
	assert len(i2Ui1_dict_consistent) == num_consistent_rotations
	return i2Ri1_dict_consistent, i2Ui1_dict_consistent


def compute_cycle_error(i2Ri1_dict, cycle_nodes, edge_i_info, edge_j_info, edge_k_info):
	"""
	Node that i1 < i2 for every valid edge
	"""
	cycle_nodes = list(cycle_nodes)
	cycle_nodes.sort()

	# (0,1) 1R0
	# (1,2) 2R1
	# (2,0) 2R0

	# 2R0 * 2R1 * 1R0 * I

	i0, i1, i2 = cycle_nodes

	i1Ri0 = i2Ri1_dict[(i0,i1)]
	i2Ri1 = i2Ri1_dict[(i1,i2)]
	i0Ri2 = i2Ri1_dict[(i0,i2)].inverse()

	i0Ri0 = i0Ri2.compose(i2Ri1).compose(i1Ri0)

	I_3x3 = Rot3()
	cycle_error = comp_utils.compute_relative_rotation_angle(I_3x3, i0Ri0)
	
	i1Ri0_euler = Rotation.from_matrix(i1Ri0.matrix()).as_euler(seq="xyz", degrees=True).tolist()
	i2Ri1_euler = Rotation.from_matrix(i2Ri1.matrix()).as_euler(seq="xyz", degrees=True).tolist()
	i0Ri2_euler = Rotation.from_matrix(i0Ri2.matrix()).as_euler(seq="xyz", degrees=True).tolist()

	euler_x = [i1Ri0_euler[0], i2Ri1_euler[0], i0Ri2_euler[0]]
	euler_y = [i1Ri0_euler[1], i2Ri1_euler[1], i0Ri2_euler[1]]
	euler_z = [i1Ri0_euler[2], i2Ri1_euler[2], i0Ri2_euler[2]]

	#import pdb; pdb.set_trace()

	rot_errors = [e_info['rotation_angular_error'] for e_info in [edge_i_info, edge_j_info, edge_k_info]]
	trans_errors = [e_info['translation_angular_error'] for e_info in [edge_i_info, edge_j_info, edge_k_info]]
	
	# average_rot_error = np.mean(rot_errors)
	# average_trans_error = np.mean(trans_errors)

	average_rot_error = np.max(rot_errors)
	average_trans_error = np.max(trans_errors)

	#if cycle_error < 10 and average_rot_error > 100:
	# print()
	# print(f"Cycle error is: {cycle_error:.1f}, w/ avg. R err {average_rot_error:.1f}, and w/ avg. t err {average_trans_error:.1f}", np.round(rot_errors,1))

	# print(f"X {euler_x[0]:.1f}, {euler_x[1]:.1f},  {euler_x[2]:.1f}")
	# print(f"Y {euler_y[0]:.1f}, {euler_y[1]:.1f},  {euler_y[2]:.1f}")
	# print(f"Z {euler_z[0]:.1f}, {euler_z[1]:.1f}, {euler_z[2]:.1f}")


	# if cycle_error < 10:
	print()
	print(f"{i0},{i1},{i2} --> Cycle error is: {cycle_error:.1f}, w/ avg. R err {average_rot_error:.1f}, and w/ avg. t err {average_trans_error:.1f}")

	print(f"X {euler_x[0]:.1f}, {euler_x[1]:.1f},  {euler_x[2]:.1f}")
	print(f"Y {euler_y[0]:.1f}, {euler_y[1]:.1f},  {euler_y[2]:.1f}")
	print(f"Z {euler_z[0]:.1f}, {euler_z[1]:.1f}, {euler_z[2]:.1f}")
	# else:
	# 	import pdb; pdb.set_trace()
	# 	assert average_rot_error > 10 or average_trans_error > 10

	return cycle_error, average_rot_error, average_trans_error


def get_quaternion_coeff_dict(R: np.ndarray):
	""" """
	qx, qy, qz, qw = Rotation.from_matrix(R.matrix()).as_quat().tolist()

	coeffs_dict = {
		"qx": qx,
		"qy": qy,
		"qz": qz,
		"qw": qw
	}
	return coeffs_dict


def test_compute_cycle_error():
	"""
	0 -> 4
	2 -> 4
	0 -> 2
	"""

	wTi0 = Pose3()
	wTi2 = Pose3()
	wTi4 = Pose3()

	i4Ri0 = wTi4.between(wTi0).rotation()
	i4Ri2 = wTi4.between(wTi2).rotation()
	i2Ri0 = wTi2.between(wTi0).rotation()

	i4Ri0 = Rot3(Rotation.from_euler("y", 95, degrees=True).as_matrix())
	i4Ri2 = Rot3(Rotation.from_euler("y", 60, degrees=True).as_matrix())
	i2Ri0 = Rot3(Rotation.from_euler("y", 30, degrees=True).as_matrix())

	cycle_nodes = [4,2,0]
	edge_i_info = {"i1": 0, "i2": 4, 'rotation_angular_error': 0, 'translation_angular_error': 0, "i2Ri1": get_quaternion_coeff_dict(i4Ri0) }
	edge_j_info = {"i1": 2, "i2": 4, 'rotation_angular_error': 0, 'translation_angular_error': 0, "i2Ri1": get_quaternion_coeff_dict(i4Ri2) }
	edge_k_info = {"i1": 0, "i2": 2, 'rotation_angular_error': 0, 'translation_angular_error': 0, "i2Ri1": get_quaternion_coeff_dict(i2Ri0) }


	cycle_error, average_rot_error, average_trans_error = compute_cycle_error(cycle_nodes, edge_i_info, edge_j_info, edge_k_info)
	assert np.isclose(cycle_error, 5)



def main():

	#edges = read_json_file("/Users/johnlambert/Downloads/gtsfm-skynet-2021-06-05/gtsfm/result_metrics/frontend_full.json")
	edges = read_json_file("/Users/johnlambert/Documents/gtsfm/result_metrics/frontend_full.json")
	num_images = 12

	i2Ui1_dict = {}
	i2Ri1_dict = {}

	two_view_reports_dict = {}

	for e_info in edges:

		i1 = e_info["i1"]
		i2 = e_info["i2"]

		if e_info["i2Ri1"]:
			coeffs_dict = e_info["i2Ri1"]
			qx,qy,qz,qw = coeffs_dict["qx"], coeffs_dict["qy"], coeffs_dict["qz"], coeffs_dict["qw"]
			i2Ri1 = Rot3(Rotation.from_quat([qx,qy,qz,qw]).as_matrix())
		else:
			i2Ri1 = None

		i2Ri1_dict[(i1,i2)] = i2Ri1

		if e_info["i2Ui1"]:
			i2Ui1 = Unit3(np.array(e_info["i2Ui1"]))
		else:
			i2Ui1 = None
		i2Ui1_dict[(i1,i2)] = i2Ui1

		report_dict = {
			"R_error_deg": e_info['rotation_angular_error'],
			"U_error_deg": e_info['translation_angular_error']
		}
		from types import SimpleNamespace
		two_view_reports_dict[(i1,i2)] = SimpleNamespace(**report_dict)

	filter_to_cycle_consistent_edges(i2Ri1_dict, i2Ui1_dict, two_view_reports_dict, visualize=True)



if __name__ == '__main__':
	main()

	#test_compute_cycle_error()

