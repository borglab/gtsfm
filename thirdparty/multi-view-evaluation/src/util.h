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

#include <unordered_map>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

typedef pcl::PointCloud<pcl::PointXYZ> PointCloud;
typedef pcl::PointCloud<pcl::PointXYZ>::Ptr PointCloudPtr;

// Make std::unordered_map work with generic tuples.
// Source: http://stackoverflow.com/questions/7110301
// Note that this is not standard-conformant.
namespace std {
  namespace {
    // Code from boost
    // Reciprocal of the golden ratio helps spread entropy
    //     and handles duplicates.
    // See Mike Seymour in magic-numbers-in-boosthash-combine:
    //     http://stackoverflow.com/questions/4948780
    
    template <class T>
    inline void hash_combine(std::size_t& seed, T const& v) {
      seed ^= std::hash<T>()(v) + 0x9e3779b9 + (seed<<6) + (seed>>2);
    }
    
    // Recursive template code derived from Matthieu M.
    template <class Tuple, size_t Index = std::tuple_size<Tuple>::value - 1>
    struct HashValueImpl {
      static void apply(size_t& seed, Tuple const& tuple) {
        HashValueImpl<Tuple, Index-1>::apply(seed, tuple);
        hash_combine(seed, std::get<Index>(tuple));
      }
    };
    
    template <class Tuple>
    struct HashValueImpl<Tuple,0> {
      static void apply(size_t& seed, Tuple const& tuple) {
        hash_combine(seed, std::get<0>(tuple));
      }
    };
  }
  
  template <typename ... TT>
  struct hash<std::tuple<TT...>> {
    size_t operator()(std::tuple<TT...> const& tt) const {
      size_t seed = 0;
      HashValueImpl<std::tuple<TT...> >::apply(seed, tt);
      return seed;
    }
  };
}

inline std::tuple<int, int, int> CalcCellCoordinates(
    const pcl::PointXYZ& point, float voxel_size_inv,
    float shift_x, float shift_y, float shift_z) {
  float voxel_x = point.x * voxel_size_inv + shift_x;
  int x = static_cast<int>(voxel_x);
  if (voxel_x - x != 0 && voxel_x < 0) {
    -- x;
  }
  
  float voxel_y = point.y * voxel_size_inv + shift_y;
  int y = static_cast<int>(voxel_y);
  if (voxel_y - y != 0 && voxel_y < 0) {
    -- y;
  }
  
  float voxel_z = point.z * voxel_size_inv + shift_z;
  int z = static_cast<int>(voxel_z);
  if (voxel_z - z != 0 && voxel_z < 0) {
    -- z;
  }
  
  return std::tuple<int, int, int>(x, y, z);
}
