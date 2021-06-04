

import json
import numpy as np
from typing import List

from colour import Color
from gtsam import Rot3, Pose3
from mayavi import mlab

import gtsfm.utils.io as io_utils
from gtsfm.common.view_frustum import ViewFrustum


def main():
	""" """
	#colmap_output_dir = "/Users/johnlambert/Downloads/crane_tower_graffiti/colmap_crane_mast_32imgs"
	#colmap_output_dir = "/Users/johnlambert/Downloads/crane_mast_with_exif/crane_mast_8imgs_v3"
	#colmap_output_dir = "/Users/johnlambert/Downloads/crane_mast_with_exif/crane_mast_8imgs_v2"
	#colmap_output_dir = "/Users/johnlambert/Downloads/crane_mast_with_exif/crane_mast_8imgs_colmap"
	colmap_output_dir = "/Users/johnlambert/Documents/gtsfm/results/ba_output"
	#colmap_output_dir = "/Users/johnlambert/Downloads/crane_mast_with_exif/ForRen_Skydio/ba_output"
	#colmap_output_dir = "/Users/johnlambert/Documents/gtsfm/tests/data/set1_lund_door/colmap_ground_truth"

	#fpath = "/Users/johnlambert/Documents/gtsfm/results/ba_input/points3D.txt"
	points_fpath = f"{colmap_output_dir}/points3D.txt"
	images_fpath = f"{colmap_output_dir}/images.txt"
	cameras_fpath = f"{colmap_output_dir}/cameras.txt"

	wTi_list, img_fnames = io_utils.read_images_txt(images_fpath)
	calibrations = io_utils.read_cameras_txt(cameras_fpath)

	if len(calibrations) == 1:
		# shared calibration!
		calibrations = calibrations * len(img_fnames)

	#import pdb; pdb.set_trace()
	point_cloud, rgb = io_utils.read_points_txt(points_fpath)

	# centralize
	#point_cloud -=

	# use robust method to estimate, due to outliers
	#centered_point_cloud = point_cloud - np.median(point_cloud, axis=0)
	ranges = np.linalg.norm(point_cloud, axis=1)
	outlier_thresh = np.percentile(ranges, 75)
	point_cloud = point_cloud[ ranges < outlier_thresh]
	rgb = rgb[ ranges < outlier_thresh]

	mean_pt = point_cloud.mean(axis=0)

	# zero-centered world
	zcworldTworld = Pose3(Rot3(np.eye(3)), -mean_pt)

	#fpath = "/Users/johnlambert/Documents/gtsfm/result_metrics/data_association_metrics.json"
	# f = open(fpath, "r")
	# data = json.load(f)
	# point_cloud = np.array(data["points_3d"])
	is_nearby = np.linalg.norm(point_cloud, axis=1) < 2000
	point_cloud = point_cloud[is_nearby]
	rgb = rgb[is_nearby]
	print(is_nearby.shape)

	#import pdb; pdb.set_trace()

	bgcolor=(1,1,1) #(0,0,0)
	fig = mlab.figure(  # type: ignore
		figure=None, bgcolor=bgcolor, fgcolor=None, engine=None, size=(1600, 1000)
	)
	draw_cameras(zcworldTworld, fig, calibrations, wTi_list)

	draw_point_cloud(zcworldTworld, fig, point_cloud, rgb)
	#draw_point_cloud_gray(point_cloud)
	mlab.show()

def draw_cameras(zcworldTworld, fig, calibrations, wTi_list: List[Pose3]):
	""" """
	#import pdb; pdb.set_trace()

	colors_arr = np.array(
		[[color_obj.rgb] for color_obj in Color("red").range_to(Color("green"), len(wTi_list) )]
	).squeeze()

	for i, (K, wTi) in enumerate(zip(calibrations, wTi_list)):
		# import pdb; pdb.set_trace()
		# points = wTi.translation().reshape(1,3)

		# draw_point_cloud(fig, points, np.array([[255,0,0]], dtype=np.uint8))

		wTi = zcworldTworld.compose(wTi)

		color = tuple(colors_arr[i].tolist())
		print(color)

		K = K.K()
		fx = K[0,0]

		# TODO: use the real image height and width

		px = K[0,2]
		py = K[1,2]

		img_w = px * 2
		img_h = py * 2
		frustum_obj = ViewFrustum(fx, img_w, img_h)

		edges_worldfr = frustum_obj.get_mesh_edges_worldframe(wTi)

		# fig = plt.figure()
		# ax = fig.gca(projection="3d")

		for edge_worldfr in edges_worldfr:

			# start and end vertices
			vs = edge_worldfr[0]
			ve = edge_worldfr[1]

			#  color, line_width, tube_radius, figure
			mlab.plot3d(  # type: ignore
				[vs[0], ve[0]],
				[vs[1], ve[1]],
				[vs[2], ve[2]],
				color=color,
				tube_radius=None,
				figure=fig,
			)


	# viz_utils.set_axes_equal(ax)
	# # uncomment line below to see viz
	# # plt.show()
	# plt.close("all")


def draw_point_cloud(zcworldTworld, fig, point_cloud: np.ndarray, rgb: np.ndarray):
	""" """
	n = point_cloud.shape[0]

	for i in range(n):
		point_cloud[i] = zcworldTworld.transformFrom(point_cloud[i])

	x, y, z = point_cloud.T
	alpha = np.ones((n,1)).astype(np.uint8) * 255 # no transparency
	rgba = np.hstack([ rgb, alpha ]).astype(np.uint8)

	pts = mlab.pipeline.scalar_scatter(x, y, z) # plot the points
	pts.add_attribute(rgba, 'colors') # assign the colors to each point
	pts.data.point_data.set_active_scalars('colors')
	g = mlab.pipeline.glyph(pts)
	g.glyph.glyph.scale_factor = 0.1 # set scaling for all the points
	g.glyph.scale_mode = 'data_scaling_off' # make all the points same size


def draw_point_cloud_gray(point_cloud: np.ndarray):

	bgcolor=(1,1,1) #(0,0,0)
	fig = mlab.figure(  # type: ignore
		figure=None, bgcolor=bgcolor, fgcolor=None, engine=None, size=(1600, 1000)
	)
	n_pts = point_cloud.shape[0]
	# draw points
	mlab.points3d(  # type: ignore
		point_cloud[:, 0],  # x
		point_cloud[:, 1],  # y
		point_cloud[:, 2],  # z
		#point_cloud[:, 2],
		scale_factor=0.1,
		mode="sphere",  # Render each point as a 'point', not as a 'sphere' or 'cube'
		colormap="copper", # "spectral",
		#color=(1,0,0), #None,  # Used a fixed (r,g,b) color instead of colormap
		figure=fig,
	)

	mlab.show()



if __name__ == '__main__':
	main()



