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

#pragma once

#include <string>
#include <vector>

#include "meshlab_project.h"
#include "util.h"

enum class AccuracyResult : uint8_t {
  kUnobserved = 0,
  kAccurate = 1,
  kInaccurate = 2
};

// Computes the accuracy of the reconstruction with respect to the given scans.
void ComputeAccuracy(
    const MeshLabMeshInfoVector& scan_infos,
    const std::vector<PointCloudPtr>& scans,
    const PointCloud& reconstruction,
    float voxel_size_inv,
    // Sorted by increasing tolerance.
    const std::vector<float>& sorted_tolerances,
    float beam_start_radius_meters,
    float tan_beam_divergence_halfangle_rad,
    // Indexed by: [tolerance_index]. Range: [0, 1].
    std::vector<float>* results,
    // Indexed by: [tolerance_index][point_index].
    std::vector<std::vector<AccuracyResult>>* point_is_accurate);

void WriteAccuracyVisualization(
    const std::string& base_path,
    const PointCloud& reconstruction,
    // Sorted by increasing tolerance.
    const std::vector<float>& sorted_tolerances,
    // Indexed by: [tolerance_index][point_index].
    const std::vector<std::vector<AccuracyResult>>& point_is_accurate);
