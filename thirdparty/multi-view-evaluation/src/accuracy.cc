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

#include "accuracy.h"

#include <Eigen/StdVector>
#include <pcl/io/ply_io.h>

// Number of cells used for the spatial access structure in inclination and
// azimuth directions.
const int kCellCountInclination = 1024;
const int kCellCountAzimuth = 2 * kCellCountInclination;

const int kGridCount = 2;
const float kGridShifts[kGridCount][3] = {{0.f, 0.f, 0.f}, {0.5f, 0.5f, 0.5f}};

// At least for Eigen::Vector3f, the aligned_allocator should not be necessary,
// but on the other hand it also shouldn't significantly hurt, and helps in not
// forgetting it.
typedef std::vector<Eigen::Matrix3f, Eigen::aligned_allocator<Eigen::Matrix3f>>
    Matrix3fVector;
typedef std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>>
    Vector3fVector;

// Accuracy results for one voxel cell.
struct AccuracyCell {
  inline AccuracyCell() : accurate_count(0), inaccurate_count(0) {}

  // Number of accurate reconstruction points within this cell.
  size_t accurate_count;

  // Number of inaccurate reconstruction points within this cell.
  size_t inaccurate_count;
};

// Modulo which works properly for negative k (in contrast to C++' % operator),
// for example, mod(-2, 10) returns 8 instead of -2.
inline int mod(int k, int n) { return ((k %= n) < 0) ? k + n : k; }

// Represents a point in spherical coordinates.
struct SphericalPoint {
  inline SphericalPoint(const Eigen::Vector3f& p) {
    radius = p.norm();
    // See: https://en.wikipedia.org/wiki/Spherical_coordinate_system
    // For the ground truth scans, X and Y are in the horizontal plane, while
    // z points up. The coordinate system is right-handed.
    azimuth = atan2(p.y(), p.x());
    inclination = acos(p.z() / radius);
  }

  float radius;
  // From -M_PI / 2 to M_PI / 2.
  float inclination;
  // From -M_PI to M_PI.
  float azimuth;
};

// Represents a point in spherical coordinates, and in addition to
// SphericalPoint also stores the normalized direction vector to the point.
struct SphericalPointAndDirection {
  inline SphericalPointAndDirection() {}
  inline SphericalPointAndDirection(const Eigen::Vector3f& p) : point(p) {
    radius = p.norm();
    // See: https://en.wikipedia.org/wiki/Spherical_coordinate_system
    // For the ground truth scans, X and Y are in the horizontal plane, while
    // z points up. The coordinate system is right-handed.
    azimuth = atan2(p.y(), p.x());
    inclination = acos(p.z() / radius);

    // Division by radius would be faster, but let's use the special-case
    // handling for zero length that normalized() presumably has.
    direction = p.normalized();
  }

  Eigen::Vector3f point;
  Eigen::Vector3f direction;

  float radius;
  // From -M_PI / 2 to M_PI / 2.
  float inclination;
  // From -M_PI to M_PI.
  float azimuth;
};

typedef std::vector<SphericalPointAndDirection> SphericalPointAndDirectionCloud;

// Stores SphericalPointAndDirection in a grid defined on azimuth and
// inclination for fast direction-based access.
class SphericalPointGrid {
  friend class SphericalPointGridIterator;

 public:
  inline SphericalPointGrid(int cell_count_azimuth, int cell_count_inclination)
      : cell_count_azimuth_(cell_count_azimuth),
        cell_extent_azimuth_(2 * M_PI / cell_count_azimuth_),
        cell_count_inclination_(cell_count_inclination),
        cell_extent_inclination_(M_PI / cell_count_inclination) {
    cells_.resize(cell_count_azimuth_ * cell_count_inclination_);
  }

  inline SphericalPointGrid(const SphericalPointGrid& other)
      : cell_count_azimuth_(other.cell_count_azimuth_),
        cell_extent_azimuth_(other.cell_extent_azimuth_),
        cell_count_inclination_(other.cell_count_inclination_),
        cell_extent_inclination_(other.cell_extent_inclination_),
        cells_(other.cells_) {}

  inline void CellCoordinatesWithoutWrap(float azimuth, float inclination,
                                         int* cell_index_azimuth,
                                         int* cell_index_inclination) const {
    float float_cell_index_azimuth = azimuth / cell_extent_azimuth_;
    *cell_index_azimuth = static_cast<int>(float_cell_index_azimuth) +
                          ((float_cell_index_azimuth < 0.f) ? (-1) : (0));

    int float_cell_index_inclination = inclination / cell_extent_inclination_;
    *cell_index_inclination =
        static_cast<int>(float_cell_index_inclination) +
        ((float_cell_index_inclination < 0.f) ? (-1) : (0));
    // Clamp.
    *cell_index_inclination = std::max(
        0, std::min(cell_count_inclination_ - 1, *cell_index_inclination));
  }

  inline int CellCoordinatesToIndex(int cell_index_azimuth,
                                    int cell_index_inclination) const {
    return cell_index_azimuth + cell_count_azimuth_ * cell_index_inclination;
  }

  // Handles clamping and wrapping around internally.
  inline int CellIndex(float azimuth, float inclination) const {
    int cell_index_azimuth;
    int cell_index_inclination;
    CellCoordinatesWithoutWrap(azimuth, inclination, &cell_index_azimuth,
                               &cell_index_inclination);

    // Wrap-around.
    cell_index_azimuth = mod(cell_index_azimuth, cell_count_azimuth_);

    return CellCoordinatesToIndex(cell_index_azimuth, cell_index_inclination);
  }

  inline const std::vector<SphericalPointAndDirection*>& cell(
      int cell_index) const {
    return cells_[cell_index];
  }
  inline const std::vector<SphericalPointAndDirection*>& cell(
      float azimuth, float inclination) const {
    return cells_[CellIndex(azimuth, inclination)];
  }
  inline std::vector<SphericalPointAndDirection*>* cell_mutable(
      int cell_index) {
    return &cells_[cell_index];
  }
  inline std::vector<SphericalPointAndDirection*>* cell_mutable(
      float azimuth, float inclination) {
    return &cells_[CellIndex(azimuth, inclination)];
  }

 private:
  int cell_count_azimuth_;
  float cell_extent_azimuth_;
  int cell_count_inclination_;
  float cell_extent_inclination_;

  // Indexed by: [cell_index_azimuth + cell_count_azimuth *
  //              cell_index_inclination][point_index] .
  // The points are not owned and must remain valid for the lifetime of this
  // object.
  std::vector<std::vector<SphericalPointAndDirection*>> cells_;
};

// Iterates over an azimuth-inclination range in a SphericalPointGrid.
class SphericalPointGridIterator {
 public:
  inline SphericalPointGridIterator(const SphericalPointGrid* grid,
                                    float azimuth, float inclination,
                                    float azimuth_angle,
                                    float inclination_angle)
      : grid_(grid) {
    grid_->CellCoordinatesWithoutWrap(
        azimuth - azimuth_angle, inclination - inclination_angle,
        &min_cell_index_azimuth_, &min_cell_index_inclination_);
    grid_->CellCoordinatesWithoutWrap(
        azimuth + azimuth_angle, inclination + inclination_angle,
        &max_cell_index_azimuth_, &max_cell_index_inclination_);

    current_cell_index_azimuth_ = min_cell_index_azimuth_ - 1;
    current_cell_index_inclination_ = min_cell_index_inclination_;
  }

  inline bool Next() {
    ++current_cell_index_azimuth_;
    if (current_cell_index_azimuth_ > max_cell_index_azimuth_) {
      current_cell_index_azimuth_ = min_cell_index_azimuth_;
      ++current_cell_index_inclination_;
      return current_cell_index_inclination_ <= max_cell_index_inclination_;
    }
    return true;
  }

  inline int cell_index() const {
    return grid_->CellCoordinatesToIndex(
        mod(current_cell_index_azimuth_, grid_->cell_count_azimuth_),
        current_cell_index_inclination_);
  }

  inline int min_cell_index_azimuth() const { return min_cell_index_azimuth_; }
  inline int max_cell_index_azimuth() const { return max_cell_index_azimuth_; }
  inline int min_cell_index_inclination() const {
    return min_cell_index_inclination_;
  }
  inline int max_cell_index_inclination() const {
    return max_cell_index_inclination_;
  }

 private:
  int current_cell_index_azimuth_;
  int current_cell_index_inclination_;

  int min_cell_index_azimuth_;
  int max_cell_index_azimuth_;
  int min_cell_index_inclination_;
  int max_cell_index_inclination_;

  const SphericalPointGrid* grid_;
};

// Classifies a point as accurate, inaccurate, or unobserved given a single
// scan point cloud. This function makes use of the following: If a
// reconstruction point is classified as accurate for some tolerance, it is also
// accurate for each higher tolerance value. Thus, the function only returns the
// index of the first (lowest) tolerance for which the point is accurate, or
// accuracy_tolerances_squared.size() if the point is not classified as accurate
// for any tolerance value. Similarly, if the point is classified as inaccurate
// for any tolerance value, it will also be classified as inaccurate for any
// smaller tolerance value. At the same time, it will be classified as accurate
// for any tolerance value larger than the highest tolerance which makes it
// classify as inaccurate. Thus, the function returns a flag which, if set,
// means that every tolerance value with an index smaller than
// first_accurate_tolerance_index classifies as inaccurate. If this flag is not
// set, the classification for these tolerance values is unobserved.
inline void ClassifyPoint(const Eigen::Vector3f& cartesian_reconstruction_point,
                          const SphericalPoint& spherical_reconstruction_point,
                          float radius_horizontal,
                          // Must be sorted in increasing order.
                          const std::vector<float>& accuracy_tolerances_squared,
                          float beam_start_radius,
                          float tan_beam_divergence_halfangle_rad,
                          const SphericalPointGrid& point_grid,
                          int* first_accurate_tolerance_index,
                          bool* inaccurate_classifications_exist) {
  *first_accurate_tolerance_index = accuracy_tolerances_squared.size();
  *inaccurate_classifications_exist = false;

  // Determine the beam radius at this distance from the scanner.
  float beam_radius =
      beam_start_radius +
      spherical_reconstruction_point.radius * tan_beam_divergence_halfangle_rad;
  float beam_radius_squared = beam_radius * beam_radius;

  // Compute the bounding box of the 2D region in (azimuth, inclination) in
  // which the scan points relevant for this reconstruction point could be:
  // Find the tangents to the sphere in vertical and horizontal direction.
  float relevancy_angle_vertical;
  float relevancy_angle_horizontal;
  if (beam_radius >= spherical_reconstruction_point.radius) {
    // Search everything.
    relevancy_angle_vertical = M_PI;
    relevancy_angle_horizontal = M_PI;
  } else {
    relevancy_angle_vertical =
        asin(beam_radius / spherical_reconstruction_point.radius);
    if (beam_radius >= radius_horizontal) {
      // Search everything horizontally.
      relevancy_angle_horizontal = M_PI;
    } else {
      relevancy_angle_horizontal = asin(beam_radius / radius_horizontal);
    }
  }

  // Intersect the bounding box with the grid cells in which the scan points are
  // stored, handling wrap-around of the spherical coordinates in the horizontal
  // direction.
  SphericalPointGridIterator it(
      &point_grid, spherical_reconstruction_point.azimuth,
      spherical_reconstruction_point.inclination, relevancy_angle_horizontal,
      relevancy_angle_vertical);
  while (it.Next()) {
    const std::vector<SphericalPointAndDirection*>& cell_points =
        point_grid.cell(it.cell_index());
    for (size_t point_index = 0, point_count = cell_points.size();
         point_index < point_count; ++point_index) {
      SphericalPointAndDirection* scan_point = cell_points[point_index];

      // Is the reconstruction point within the beam volume? (Checked by testing
      // whether the scan ray is closer than beam_radius to the reconstruction
      // point).
      float signed_distance_along_ray =
          scan_point->direction.dot(cartesian_reconstruction_point);
      if (signed_distance_along_ray < 0) {
        // Treat points on the opposite side of the scan ray as unobserved.
        continue;
      }
      Eigen::Vector3f closest_point_on_scan_ray =
          signed_distance_along_ray * scan_point->direction;
      float scan_ray_distance_squared =
          (cartesian_reconstruction_point - closest_point_on_scan_ray)
              .squaredNorm();
      if (scan_ray_distance_squared <= beam_radius_squared) {
        // Is the reconstruction point within the region for accurate
        // classification (i.e., closer to the scan point than the evaluation
        // threshold)? In this case, early exit with accurate classification.
        float distance_from_scan_point_squared =
            (scan_point->point - cartesian_reconstruction_point).squaredNorm();
        for (size_t tolerance_index = 0;
             tolerance_index < accuracy_tolerances_squared.size() &&
             static_cast<int>(tolerance_index) <
                 *first_accurate_tolerance_index;
             ++tolerance_index) {
          if (distance_from_scan_point_squared <=
              accuracy_tolerances_squared[tolerance_index]) {
            *first_accurate_tolerance_index = tolerance_index;
            if (tolerance_index == 0) {
              // Early exit.
              return;
            }
            break;
          }
        }

        // Is the reconstruction point in front of the scan point? In this case,
        // remember that inaccurate classifications may exist.
        if (signed_distance_along_ray < scan_point->radius) {
          *inaccurate_classifications_exist = true;
        }
      }
    }
  }
}

void ComputeAccuracy(
    const MeshLabMeshInfoVector& scan_infos,
    const std::vector<PointCloudPtr>& scans, const PointCloud& reconstruction,
    float voxel_size_inv,
    // Sorted by increasing tolerance.
    const std::vector<float>& sorted_tolerances, float beam_start_radius_meters,
    float tan_beam_divergence_halfangle_rad,
    // Indexed by: [tolerance_index]. Range: [0, 1].
    std::vector<float>* results,
    // Indexed by: [tolerance_index][point_index].
    std::vector<std::vector<AccuracyResult>>* point_is_accurate) {
  bool output_point_results = point_is_accurate != nullptr;

  size_t scan_count = scans.size();
  size_t tolerances_count = sorted_tolerances.size();

  // Compute squared tolerances.
  std::vector<float> sorted_tolerances_squared(tolerances_count);
  for (size_t tolerance_index = 0; tolerance_index < tolerances_count;
       ++tolerance_index) {
    sorted_tolerances_squared[tolerance_index] =
        sorted_tolerances[tolerance_index] * sorted_tolerances[tolerance_index];
  }

  // Cache scan_T_global transformations in matrix+vector form for fast
  // global-to-scan transforming.
  Matrix3fVector scans_R_global(scan_count);
  Vector3fVector scans_T_global(scan_count);
  for (size_t scan_index = 0; scan_index < scan_count; ++scan_index) {
    const Eigen::Matrix4f& global_T_mesh = scan_infos[scan_index].global_T_mesh;

    // Rotation matrix inverse is its transpose.
    scans_R_global[scan_index] = global_T_mesh.block<3, 3>(0, 0).transpose();

    // Translation of inverse transformation is -(R^(-1) * t):
    //   y = R * x + t
    //   y - t = R * x
    //   R^(-1) * (y - t) = x
    //   x = R^(-1) * y - (R^(-1) * t)
    scans_T_global[scan_index] =
        scans_R_global[scan_index] * (-1 * global_T_mesh.block<3, 1>(0, 3));
  }

  // Transform all scan points to spherical coordinates, and sort them into grid
  // cells defined on the spherical coordinates.
  std::vector<SphericalPointAndDirectionCloud> spherical_clouds(scan_count);
  std::vector<std::shared_ptr<SphericalPointGrid>> point_grids(scan_count);
  for (size_t scan_index = 0; scan_index < scan_count; ++scan_index) {
    point_grids[scan_index].reset(
        new SphericalPointGrid(kCellCountAzimuth, kCellCountInclination));

    const PointCloud& cartesian_cloud = *scans[scan_index];
    SphericalPointAndDirectionCloud* spherical_cloud =
        &spherical_clouds[scan_index];
    spherical_cloud->resize(cartesian_cloud.size());
    for (size_t p = 0; p < cartesian_cloud.size(); ++p) {
      const pcl::PointXYZ& cartesian_point = cartesian_cloud.at(p);
      SphericalPointAndDirection* spherical_point = &spherical_cloud->at(p);
      *spherical_point =
          SphericalPointAndDirection(cartesian_point.getVector3fMap());

      point_grids[scan_index]
          ->cell_mutable(spherical_point->azimuth, spherical_point->inclination)
          ->push_back(spherical_point);
    }
  }

  // Prepare point_is_accurate, if requested.
  if (output_point_results) {
    point_is_accurate->resize(tolerances_count);
    for (size_t tolerance_index = 0; tolerance_index < tolerances_count;
         ++tolerance_index) {
      point_is_accurate->at(tolerance_index)
          .resize(reconstruction.size(), AccuracyResult::kUnobserved);
    }
  }

  // Differently shifted voxel grids.
  // Indexed by: [map_index][CalcCellCoordinates(...)][tolerance_index].
  std::unordered_map<std::tuple<int, int, int>, std::vector<AccuracyCell>>
      cell_maps[kGridCount];

  // Loop over the reconstruction points.
  for (size_t point_index = 0, size = reconstruction.size(); point_index < size;
       ++point_index) {
    const pcl::PointXYZ& point = reconstruction.at(point_index);

    // Find the voxels for this reconstruction point.
    std::vector<AccuracyCell>* cell_vectors[kGridCount];
    for (int grid_index = 0; grid_index < kGridCount; ++grid_index) {
      std::vector<AccuracyCell>* cell_vector =
          &cell_maps[grid_index][CalcCellCoordinates(
              point, voxel_size_inv, kGridShifts[grid_index][0],
              kGridShifts[grid_index][1], kGridShifts[grid_index][2])];
      if (cell_vector->empty()) {
        cell_vector->resize(tolerances_count);
      }
      cell_vectors[grid_index] = cell_vector;
    }

    int aggregate_first_accurate_tolerance_index =
        static_cast<int>(sorted_tolerances_squared.size());
    bool aggregate_inaccurate_classifications_exist = false;

    // Loop over the ground truth point clouds and classify the reconstruction
    // point for each ground truth scan. Then compute an aggregate result
    // from the individual ones: If the point is accurate given at least one
    // scan, the aggregate result is also accurate. Otherwise, if the point is
    // inaccurate given at least one scan, the aggregate result is inaccurate.
    // Otherwise, it is unobserved.
    for (size_t scan_index = 0; scan_index < scan_count; ++scan_index) {
      // Transform reconstruction point into the cloud frame.
      Eigen::Vector3f cartesian_reconstruction_point =
          scans_R_global[scan_index] * point.getVector3fMap() +
          scans_T_global[scan_index];

      // Convert it to spherical coordinates.
      SphericalPoint spherical_reconstruction_point(
          cartesian_reconstruction_point);

      // Classify it.
      SphericalPointGrid* point_grid = point_grids[scan_index].get();
      int first_accurate_tolerance_index;
      bool inaccurate_classifications_exist;
      float radius_horizontal = sqrtf(cartesian_reconstruction_point.x() *
                                          cartesian_reconstruction_point.x() +
                                      cartesian_reconstruction_point.y() *
                                          cartesian_reconstruction_point.y());
      ClassifyPoint(cartesian_reconstruction_point,
                    spherical_reconstruction_point, radius_horizontal,
                    sorted_tolerances_squared, beam_start_radius_meters,
                    tan_beam_divergence_halfangle_rad, *point_grid,
                    &first_accurate_tolerance_index,
                    &inaccurate_classifications_exist);

      // Merge into aggregate result and exit early if the point is accurate for
      // all tolerances.
      aggregate_inaccurate_classifications_exist |=
          inaccurate_classifications_exist;
      if (first_accurate_tolerance_index <
          aggregate_first_accurate_tolerance_index) {
        aggregate_first_accurate_tolerance_index =
            first_accurate_tolerance_index;
        if (aggregate_first_accurate_tolerance_index == 0) {
          break;
        }
      }
    }

    // Aggregate accurate count.
    for (int tolerance_index = aggregate_first_accurate_tolerance_index;
         tolerance_index < static_cast<int>(tolerances_count);
         ++tolerance_index) {
      for (int grid_index = 0; grid_index < kGridCount; ++grid_index) {
        ++cell_vectors[grid_index]->at(tolerance_index).accurate_count;
      }
      if (output_point_results) {
        point_is_accurate->at(tolerance_index)[point_index] =
            AccuracyResult::kAccurate;
      }
    }
    // Aggregate inaccurate count or unobserved count.
    if (aggregate_inaccurate_classifications_exist) {
      for (int tolerance_index = aggregate_first_accurate_tolerance_index - 1;
           tolerance_index >= 0; --tolerance_index) {
        for (int grid_index = 0; grid_index < kGridCount; ++grid_index) {
          ++cell_vectors[grid_index]->at(tolerance_index).inaccurate_count;
        }
        if (output_point_results) {
          point_is_accurate->at(tolerance_index)[point_index] =
              AccuracyResult::kInaccurate;
        }
      }
    } else {
      if (output_point_results) {
        for (int tolerance_index = aggregate_first_accurate_tolerance_index - 1;
             tolerance_index >= 0; --tolerance_index) {
          point_is_accurate->at(tolerance_index)[point_index] =
              AccuracyResult::kUnobserved;
        }
      }
    }
  }

  // Average results over all cells and fill the results vector.
  std::vector<double> accuracy_sum(tolerances_count, 0.0);
  std::vector<size_t> valid_cell_count(tolerances_count, 0);
  for (int grid_index = 0; grid_index < kGridCount; ++grid_index) {
    for (auto it = cell_maps[grid_index].cbegin(),
              end = cell_maps[grid_index].cend();
         it != end; ++it) {
      const std::vector<AccuracyCell>& cell_vector = it->second;
      for (size_t tolerance_index = 0; tolerance_index < tolerances_count;
           ++tolerance_index) {
        const AccuracyCell& cell = cell_vector[tolerance_index];
        size_t valid_point_count = cell.accurate_count + cell.inaccurate_count;
        if (valid_point_count > 0) {
          accuracy_sum[tolerance_index] +=
              cell.accurate_count / (1.0f * valid_point_count);
          ++valid_cell_count[tolerance_index];
        }
      }
    }
  }

  results->resize(tolerances_count);
  for (size_t tolerance_index = 0; tolerance_index < tolerances_count;
       ++tolerance_index) {
    float* accuracy = &results->at(tolerance_index);
    if (valid_cell_count[tolerance_index] == 0) {
      *accuracy = 0;
    } else {
      *accuracy =
          accuracy_sum[tolerance_index] / valid_cell_count[tolerance_index];
    }
  }
}

void WriteAccuracyVisualization(
    const std::string& base_path, const PointCloud& reconstruction,
    // Sorted by increasing tolerance.
    const std::vector<float>& sorted_tolerances,
    // Indexed by: [tolerance_index][point_index].
    const std::vector<std::vector<AccuracyResult>>& point_is_accurate) {
  pcl::PointCloud<pcl::PointXYZRGB> accuracy_visualization;
  accuracy_visualization.resize(reconstruction.size());

  // Set visualization point positions (once for all tolerances).
  for (size_t i = 0; i < reconstruction.size(); ++i) {
    accuracy_visualization.at(i).getVector3fMap() =
        reconstruction.at(i).getVector3fMap();
  }

  // Loop over all tolerances, set visualization point colors accordingly and
  // save the point clouds.
  for (size_t tolerance_index = 0; tolerance_index < sorted_tolerances.size();
       ++tolerance_index) {
    const std::vector<AccuracyResult>& point_is_accurate_for_tolerance =
        point_is_accurate[tolerance_index];

    for (size_t point_index = 0; point_index < accuracy_visualization.size();
         ++point_index) {
      pcl::PointXYZRGB* point = &accuracy_visualization.at(point_index);
      if (point_is_accurate_for_tolerance[point_index] ==
          AccuracyResult::kAccurate) {
        // Green: accurate points.
        point->r = 0;
        point->g = 255;
        point->b = 0;
      } else if (point_is_accurate_for_tolerance[point_index] ==
                 AccuracyResult::kInaccurate) {
        // Red: inaccurate points.
        point->r = 255;
        point->g = 0;
        point->b = 0;
      } else if (point_is_accurate_for_tolerance[point_index] ==
                 AccuracyResult::kUnobserved) {
        // Blue: unobserved points.
        point->r = 0;
        point->g = 0;
        point->b = 255;
      }
    }

    std::ostringstream file_path;
    file_path << base_path << ".tolerance_"
              << sorted_tolerances[tolerance_index] << ".ply";
    pcl::io::savePLYFileBinary(file_path.str(), accuracy_visualization);
  }
}
