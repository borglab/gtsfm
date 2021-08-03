// Copyright 2017 Thomas Schöps
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its contributors
//    may be used to endorse or promote products derived from this software
//    without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#include <Eigen/Core>
#include <boost/filesystem.hpp>
#include <pcl/console/parse.h>
#include <pcl/io/ply_io.h>

#include "accuracy.h"
#include "completeness.h"
#include "meshlab_project.h"
#include "util.h"

const float kDegToRadFactor = M_PI / 180.0;

// Return codes of the program:
// 0: Success.
// 1: System failure (e.g., due to wrong parameters given).
// 2: Reconstruction file input failure (PLY file cannot be found or read).
enum class ReturnCodes {
  kSuccess = 0,
  kSystemFailure = 1,
  kReconstructionFileInputFailure = 2
};

int main(int argc, char** argv) {
  pcl::console::setVerbosityLevel(pcl::console::L_ALWAYS);

  // Parse arguments.
  std::string reconstruction_ply_path;
  pcl::console::parse_argument(argc, argv, "--reconstruction_ply_path",
                               reconstruction_ply_path);
  std::string ground_truth_mlp_path;
  pcl::console::parse_argument(argc, argv, "--ground_truth_mlp_path",
                               ground_truth_mlp_path);
  std::vector<float> tolerances;
  pcl::console::parse_x_arguments(argc, argv, "--tolerances", tolerances);
  float voxel_size = 0.01f;
  pcl::console::parse_argument(argc, argv, "--voxel_size", voxel_size);
  float beam_start_radius_meters = 0.5 * 0.00225;
  pcl::console::parse_argument(argc, argv, "--beam_start_radius_meters",
                               beam_start_radius_meters);
  float beam_divergence_halfangle_deg = 0.011;
  pcl::console::parse_argument(argc, argv, "--beam_divergence_halfangle_deg",
                               beam_divergence_halfangle_deg);
  std::string completeness_cloud_output_path;
  pcl::console::parse_argument(argc, argv, "--completeness_cloud_output_path",
                               completeness_cloud_output_path);
  std::string accuracy_cloud_output_path;
  pcl::console::parse_argument(argc, argv, "--accuracy_cloud_output_path",
                               accuracy_cloud_output_path);

  // Validate arguments.
  std::stringstream errors;
  if (tolerances.empty()) {
    errors << "The --tolerances parameter must be given as a list of"
           << " non-negative evaluation tolerance values, separated by"
           << " commas." << std::endl;
  }
  if (reconstruction_ply_path.empty()) {
    errors << "The --reconstruction_ply_path parameter must be given."
           << std::endl;
  }
  if (ground_truth_mlp_path.empty()) {
    errors << "The --ground_truth_mlp_path parameter must be given."
           << std::endl;
  }
  if (voxel_size <= 0.f) {
    errors << "The voxel size must be positive." << std::endl;
  }

  if (!errors.str().empty()) {
    std::cerr << "Usage example: " << argv[0]
              << " --tolerances 0.1,0.2 --reconstruction_ply_path "
                 "path/to/reconstruction.ply --ground_truth_mlp_path "
                 "path/to/ground-truth.mlp"
              << std::endl << std::endl;
    std::cerr << errors.str() << std::endl;
    return static_cast<int>(ReturnCodes::kSystemFailure);
  }

  // Process arguments.
  std::sort(tolerances.begin(), tolerances.end());
  float tan_beam_divergence_halfangle_rad =
      tan(kDegToRadFactor * beam_divergence_halfangle_deg);
  float voxel_size_inv = 1.0 / voxel_size;

  // Load the ground truth point cloud poses from the MeshLab project file.
  MeshLabMeshInfoVector scan_infos;
  if (!ReadMeshLabProject(ground_truth_mlp_path, &scan_infos)) {
    std::cerr << "Cannot read scan poses from " << ground_truth_mlp_path
              << std::endl;
    return static_cast<int>(ReturnCodes::kSystemFailure);
  }

  // Validate the scan transformations (partly: not checking that the top-left
  // 3x3 block is a rotation matrix).
  for (size_t scan_index = 0; scan_index < scan_infos.size(); ++scan_index) {
    const MeshLabProjectMeshInfo& info = scan_infos[scan_index];
    if (info.global_T_mesh(3, 0) != 0 || info.global_T_mesh(3, 1) != 0 ||
        info.global_T_mesh(3, 2) != 0 || info.global_T_mesh(3, 3) != 1) {
      std::cerr << "Error: Last row in a scan's transformation matrix is not"
                   " (0, 0, 0, 1)."
                << std::endl;
      return static_cast<int>(ReturnCodes::kSystemFailure);
    }
  }

  // Load the reconstruction point cloud.
  std::cout << "Loading reconstruction: " << reconstruction_ply_path
            << std::endl;
  PointCloudPtr reconstruction(new PointCloud());
  if (pcl::io::loadPLYFile(reconstruction_ply_path, *reconstruction) < 0) {
    std::cerr << "Cannot read reconstruction file." << std::endl;
    return static_cast<int>(ReturnCodes::kReconstructionFileInputFailure);
  }

  // Load the ground truth scan point clouds.
  std::vector<PointCloudPtr> scans;
  for (const MeshLabProjectMeshInfo& scan_info : scan_infos) {
    // Get absolute or compose relative path.
    std::string file_path =
        (scan_info.filename.empty() || scan_info.filename[0] == '/')
            ? scan_info.filename
            : (boost::filesystem::path(ground_truth_mlp_path).parent_path() /
               scan_info.filename)
                  .string();

    std::cout << "Loading scan: " << file_path << std::endl;
    PointCloudPtr point_cloud(new PointCloud());
    if (pcl::io::loadPLYFile(file_path, *point_cloud) < 0) {
      std::cerr << "Cannot read scan file." << std::endl;
      return static_cast<int>(ReturnCodes::kSystemFailure);
    }

    scans.push_back(point_cloud);
  }

  // Determine completeness.
  std::cout << "Computing completeness" << std::endl;

  // Indexed by: [tolerance_index].
  std::vector<float> completeness_results;
  // Indexed by: [tolerance_index][scan_point_index].
  std::vector<std::vector<bool>> point_is_complete;
  bool output_point_completeness = !completeness_cloud_output_path.empty();
  ComputeCompleteness(scan_infos, scans, reconstruction, voxel_size_inv,
                      tolerances, &completeness_results,
                      output_point_completeness ? &point_is_complete : nullptr);

  // Write completeness visualization, if requested.
  if (output_point_completeness) {
    boost::filesystem::create_directories(
        boost::filesystem::path(completeness_cloud_output_path).parent_path());
    WriteCompletenessVisualization(completeness_cloud_output_path, scan_infos,
                                   scans, tolerances, point_is_complete);
  }

  // Determine accuracy.
  std::cout << "Computing accuracy" << std::endl;

  // Indexed by: [tolerance_index].
  std::vector<float> accuracy_results;
  // Indexed by: [tolerance_index][scan_point_index].
  std::vector<std::vector<AccuracyResult>> point_is_accurate;
  bool output_point_accuracy = !accuracy_cloud_output_path.empty();
  ComputeAccuracy(scan_infos, scans, *reconstruction, voxel_size_inv,
                  tolerances, beam_start_radius_meters,
                  tan_beam_divergence_halfangle_rad, &accuracy_results,
                  output_point_accuracy ? &point_is_accurate : nullptr);

  // Write accuracy visualization, if requested.
  if (output_point_accuracy) {
    boost::filesystem::create_directories(
        boost::filesystem::path(accuracy_cloud_output_path).parent_path());
    WriteAccuracyVisualization(accuracy_cloud_output_path, *reconstruction,
                               tolerances, point_is_accurate);
  }

  // Output results.
  std::cout << "Tolerances: ";
  for (size_t tolerance_index = 0; tolerance_index < tolerances.size();
       ++tolerance_index) {
    std::cout << tolerances[tolerance_index];
    if (tolerance_index < tolerances.size() - 1) {
      std::cout << " ";
    }
  }
  std::cout << std::endl;

  std::cout << "Completenesses: ";
  for (size_t tolerance_index = 0; tolerance_index < tolerances.size();
       ++tolerance_index) {
    std::cout << completeness_results[tolerance_index];
    if (tolerance_index < tolerances.size() - 1) {
      std::cout << " ";
    }
  }
  std::cout << std::endl;

  std::cout << "Accuracies: ";
  for (size_t tolerance_index = 0; tolerance_index < tolerances.size();
       ++tolerance_index) {
    std::cout << accuracy_results[tolerance_index];
    if (tolerance_index < tolerances.size() - 1) {
      std::cout << " ";
    }
  }
  std::cout << std::endl;

  // Balanced F-score putting the same weight on accuracy and completeness, see:
  // https://en.wikipedia.org/wiki/F1_score
  std::cout << "F1-scores: ";
  for (size_t tolerance_index = 0; tolerance_index < tolerances.size();
       ++tolerance_index) {
    float precision = accuracy_results[tolerance_index];
    float recall = completeness_results[tolerance_index];
    float f1_score = (precision <= 0 && recall <= 0)
                         ? 0
                         : (2 * (precision * recall) / (precision + recall));
    std::cout << f1_score;
    if (tolerance_index < tolerances.size() - 1) {
      std::cout << " ";
    }
  }
  std::cout << std::endl;

  return static_cast<int>(ReturnCodes::kSuccess);
}
