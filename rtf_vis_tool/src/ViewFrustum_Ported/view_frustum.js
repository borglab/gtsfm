var nj = require('numjs');
var SE3 = require('./se3.js');

const DEFAULT_FRUSTUM_RAY_LENGTH = 0.5;  //meters, arbitrary

module.exports = class ViewFrustum {
    //Generates edges of a 5-face mesh for drawing pinhole camera in 3d

    constructor(fx, img_w, img_h) {
        /*
            Args:
                fx: focal length in x-direction, assuming square pixels (fx == fy)
                img_w: image width (in pixels)
                img_h: image height (in pixels)
                frustum_ray_len: extent to which extend frustum rays away from optical center
                    (increase length for large-scale scenes to make frustums visible)
        */
       this.fx_ = fx;
       this.img_w_ = img_w;
       this.img_h_ = img_h;
       this.frustum_ray_len_ = DEFAULT_FRUSTUM_RAY_LENGTH;
    }

    normalize_ray_dirs(ray_dirs) {
        var v0_mag = Math.sqrt(Math.pow(ray_dirs.get(0,0),2) + Math.pow(ray_dirs.get(0,1),2) + Math.pow(ray_dirs.get(0,2),2));
        var v1_mag = Math.sqrt(Math.pow(ray_dirs.get(1,0),2) + Math.pow(ray_dirs.get(1,1),2) + Math.pow(ray_dirs.get(1,2),2));
        var v2_mag = Math.sqrt(Math.pow(ray_dirs.get(2,0),2) + Math.pow(ray_dirs.get(2,1),2) + Math.pow(ray_dirs.get(2,2),2));
        var v3_mag = Math.sqrt(Math.pow(ray_dirs.get(3,0),2) + Math.pow(ray_dirs.get(3,1),2) + Math.pow(ray_dirs.get(3,2),2));
        var v4_mag = Math.sqrt(Math.pow(ray_dirs.get(4,0),2) + Math.pow(ray_dirs.get(4,1),2) + Math.pow(ray_dirs.get(4,2),2));

        ray_dirs.set(0, 0, (ray_dirs.get(0,0) / v0_mag));
        ray_dirs.set(0, 1, (ray_dirs.get(0,1) / v0_mag));
        ray_dirs.set(0, 2, (ray_dirs.get(0,2) / v0_mag));
        ray_dirs.set(1, 0, (ray_dirs.get(1,0) / v1_mag));
        ray_dirs.set(1, 1, (ray_dirs.get(1,1) / v1_mag));
        ray_dirs.set(1, 2, (ray_dirs.get(1,2) / v1_mag));
        ray_dirs.set(2, 0, (ray_dirs.get(2,0) / v2_mag));
        ray_dirs.set(2, 1, (ray_dirs.get(2,1) / v2_mag));
        ray_dirs.set(2, 2, (ray_dirs.get(2,2) / v2_mag));
        ray_dirs.set(3, 0, (ray_dirs.get(3,0) / v3_mag));
        ray_dirs.set(3, 1, (ray_dirs.get(3,1) / v3_mag));
        ray_dirs.set(3, 2, (ray_dirs.get(3,2) / v3_mag));
        ray_dirs.set(4, 0, (ray_dirs.get(4,0) / v4_mag));
        ray_dirs.set(4, 1, (ray_dirs.get(4,1) / v4_mag));
        ray_dirs.set(4, 2, (ray_dirs.get(4,2) / v4_mag));

        return ray_dirs;
    }

    compute_pixel_ray_directions_vectorized(uv, fx, img_w, img_h) {
        /*
            Given (u,v) coordinates and intrinsics, generate pixels rays in cam. coord frame
            Assume +z points out of the camera, +y is downwards, and +x is across the imager.
            Args:
                uv: array of shape (N,2) with (u,v) coordinates
                fx: focal length in x-direction, assuming square pixels (fx == fy)
                img_w: image width (in pixels)
                img_h: image height (in pixels)
            Returns:
                ray_dirs: Array of shape (N,3) with ray directions in camera frame
        */

        //assuming principal point at center of images now
        var px = img_w / 2;
        var py = img_h / 2;

        var num_rays = uv.shape[0];

        //broadcase (1,2) across (N,2) uv array
        var centers_broadcasted = nj.array([
            [px,py],
            [px,py],
            [px,py],
            [px,py],
            [px,py]
        ]);
        var center_offsets = uv.subtract(centers_broadcasted);
        var ray_dirs = nj.zeros([num_rays, 3]);

        ray_dirs.set(0,0, center_offsets.get(0,0));
        ray_dirs.set(0,1, center_offsets.get(0,1));
        ray_dirs.set(1,0, center_offsets.get(1,0));
        ray_dirs.set(1,1, center_offsets.get(1,1));
        ray_dirs.set(2,0, center_offsets.get(2,0));
        ray_dirs.set(2,1, center_offsets.get(2,1));
        ray_dirs.set(3,0, center_offsets.get(3,0));
        ray_dirs.set(3,1, center_offsets.get(3,1));
        ray_dirs.set(4,0, center_offsets.get(4,0));
        ray_dirs.set(4,1, center_offsets.get(4,1));
        ray_dirs.set(0,2, this.fx_);
        ray_dirs.set(1,2, this.fx_);
        ray_dirs.set(2,2, this.fx_);
        ray_dirs.set(3,2, this.fx_);
        ray_dirs.set(4,2, this.fx_);

        ray_dirs = this.normalize_ray_dirs(ray_dirs); //TODO
        return ray_dirs
    }

    get_frustum_vertices_camfr() {
        /*
            Obtain 3d positions of all 5 frustum vertices in the camera frame
          (x,y,z)               (x,y,z)                (x,y,z)              (x,y,z)
              \\=================//                      \\                   //
               \\               //                        \\ 1-------------2 //
        (-w/2,-h/2,fx)       (w/2,-h/2,fx)                 \\| IMAGE PLANE |//
                 1-------------2                             |             |/
                 |\\         //| IMAGE PLANE  (-w/2, h/2,fx) 4-------------3 (w/2, h/2,fx)
                 | \\       // | IMAGE PLANE                  \\         //
                 4--\\-----//--3                               \\       //
                     \\   //                                    \\     //
                      \\ //                                      \\   //
                        O PINHOLE                                 \\ //
                                                                    O PINHOLE
        */

        var uv = nj.array([
            [Math.floor(this.img_w_ / 2), Math.floor(this.img_h_ / 2)],  //v0 = optical center
            [0, 0],                              //v1 = top-left
            [this.img_w_ - 1, 0],                //v2 = top-right
            [this.img_w_ - 1, this.img_h_ - 1],  //v3 = bottom-right
            [0, this.img_h_ - 1],                //v4 = bottom-left
        ]);
        var ray_dirs = this.compute_pixel_ray_directions_vectorized(uv, this.fx, this.img_w_, this.img_h_);
        // var v0 = ray_dirs[0] * 0;
        // var v1 = ray_dirs[1] * this.frustum_ray_len_;
        // var v2 = ray_dirs[2] * this.frustum_ray_len_;
        // var v3 = ray_dirs[3] * this.frustum_ray_len_;
        // var v4 = ray_dirs[4] * this.frustum_ray_len_;
        // return [v0, v1, v2, v3, v4];
        var scaled_ray_dirs = this.calculate_scaled_ray_dirs(ray_dirs, this.frustum_ray_len_);
        return scaled_ray_dirs
    }

    get_mesh_vertices_worldframe(wTc) {
        /*Return 8 edges defining the frustum mesh, in the world/global frame.
            Args:
                wTc: camera pose in world frame
            Returns:
                edges_worldfr: array of shape (8,3,2) representing 8 polylines in world frame
        */
       var vList = this.get_frustum_vertices_camfr();
       var v0 = vList[0].reshape(1,3);
       var v1 = vList[1].reshape(1,3);
       var v2 = vList[2].reshape(1,3);
       var v3 = vList[3].reshape(1,3);
       var v4 = vList[4].reshape(1,3);

       var v0_worldfr = wTc.transform_point_cloud(v0);
       var v1_worldfr = wTc.transform_point_cloud(v1);
       var v2_worldfr = wTc.transform_point_cloud(v2);
       var v3_worldfr = wTc.transform_point_cloud(v3);
       var v4_worldfr = wTc.transform_point_cloud(v4);

       return [v0_worldfr, v1_worldfr, v2_worldfr, v3_worldfr, v4_worldfr];
    }

    calculate_scaled_ray_dirs(ray_dirs, frustum_ray_len) {
        var v0_scaled = nj.array([ray_dirs.get(0,0) * 0, 
                                ray_dirs.get(0,1) * 0, 
                                ray_dirs.get(0,2) * 0]);
        var v1_scaled = nj.array([ray_dirs.get(1,0) * frustum_ray_len,
                                ray_dirs.get(1,1) * frustum_ray_len,
                                ray_dirs.get(1,2) * frustum_ray_len]);
        var v2_scaled = nj.array([ray_dirs.get(2,0) * frustum_ray_len,
                                ray_dirs.get(2,1) * frustum_ray_len,
                                ray_dirs.get(2,2) * frustum_ray_len]);
        var v3_scaled = nj.array([ray_dirs.get(3,0) * frustum_ray_len,
                                ray_dirs.get(3,1) * frustum_ray_len,
                                ray_dirs.get(3,2) * frustum_ray_len]);
        var v4_scaled = nj.array([ray_dirs.get(4,0) * frustum_ray_len,
                                ray_dirs.get(4,1) * frustum_ray_len,
                                ray_dirs.get(4,2) * frustum_ray_len]);
        return [v0_scaled, v1_scaled, v2_scaled, v3_scaled, v4_scaled];
    }
}



// var rotation = nj.array([
//     [4,3,2],
//     [10,9,8],
//     [6,5,4]
// ]);
// var translation = nj.array([99,98,97]);
// sampleVF = new ViewFrustum(10, 200, 100);
// wTc = new SE3(rotation, translation);
// output = sampleVF.get_mesh_vertices_worldframe(wTc);
// console.log(output[4]);