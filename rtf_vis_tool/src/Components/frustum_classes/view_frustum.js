/*Contains functions to transform camera instrinsic variables into a frustum consisting of 5 vertices.

Author: Adi Singh
*/

// Third-Party Package Imports.
var nj = require('numjs');

class ViewFrustum {
    // Generates vertices of a 5-face mesh for drawing a pinhole camera in 3d.

    constructor(fx, img_w, img_h, ray_len) {
        /*Initializes the ViewFrustum Object.

        Args:
            fx (float): Focal length in x-direction (in pixels).
            img_w (float): Image width (in pixels).
            img_h (float): Image height (in pixels).
            ray_len (float): Extent to which frustum rays extend away from optical center.
        */

        this.fx_ = fx;
        this.img_w_ = img_w;
        this.img_h_ = img_h;
        this.frustum_ray_len_ = ray_len;
    }

    normalize_ray_dirs(ray_dirs) {
        /*Normalizes the 5 ray directions of the frustum vertices to be of unit length.

        Args:
            ray_dirs: Array of shape (5,3) representing ray directions in camera frame.
        Returns:
            An array of shape (5,3) with normalized ray directions.
        */

        for (var row = 0; row < ray_dirs.shape[0]; row++) {
            var vector_mag = Math.sqrt(Math.pow(ray_dirs.get(row,0),2) + Math.pow(ray_dirs.get(row,1),2) + Math.pow(ray_dirs.get(row,2),2));
            for (var col = 0; col < ray_dirs.shape[1]; col++) {
                ray_dirs.set(row, col, (ray_dirs.get(row,col)/vector_mag));
            }
        }
        return ray_dirs;
    }

    compute_pixel_ray_directions_vectorized(uv, img_w, img_h) {
        /*Given (u,v) coordinates and intrinsics, generate pixels rays in camera coord frame. Assume +z points out of
        the camera, +y is downwards, and +x is across the imager.

        Args:
            uv: Array of shape (5,2) with (u,v) coordinates.
            img_w (float): Image width (in pixels).
            img_h (height): Image height (in pixels).
        Returns:
            ray_dirs: Array of shape (5,3) with normalized ray vectors in camera frame
        */

        // Assume principal point is at center of images.
        var px = img_w / 2;
        var py = img_h / 2;

        //uv - [px,py] gives each vertex's pixel offset from the center of image plane.
        var center_offsets = nj.zeros([5,2]);
        for (var row = 0; row < 5; row++) {
            center_offsets.set(row, 0, uv.get(row, 0) - px);
            center_offsets.set(row, 1, uv.get(row, 1) - py);
        }

        var fx_broadcasted = nj.array([
            [this.fx_],
            [this.fx_],
            [this.fx_],
            [this.fx_],
            [this.fx_]
        ]);
        var ray_dirs = nj.concatenate(center_offsets, fx_broadcasted);
        ray_dirs = this.normalize_ray_dirs(ray_dirs);
        return ray_dirs
    }

    get_frustum_vertices_camfr() {
        /*Obtain 3d positions of all 5 frustum vertices in the camera frame

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
            
        Returns:
            Array, shape (5,3), of frustum vertex coordinates in the camera frame.                                                                
        */

        var uv = nj.array([
            [Math.floor(this.img_w_ / 2), Math.floor(this.img_h_ / 2)],  //v0 = optical center
            [0, 0],                              //v1 = top-left
            [this.img_w_ - 1, 0],                //v2 = top-right
            [this.img_w_ - 1, this.img_h_ - 1],  //v3 = bottom-right
            [0, this.img_h_ - 1],                //v4 = bottom-left
        ]);

        var ray_dirs = this.compute_pixel_ray_directions_vectorized(uv, this.img_w_, this.img_h_);
        var scaled_ray_dirs = this.calculate_scaled_ray_dirs(ray_dirs, this.frustum_ray_len_);
        return scaled_ray_dirs
    }

    get_mesh_vertices_worldframe(cTw) {
        /*Obtains 5 vertices defining the frustum mesh, in the world/global frame.

        Args:
            cTw (SE3): camera extrinsics matrix
        Returns:
            An array, length 5, where each entry represents the 3d coordinate of a frustum vertex.
        */
        var vList = this.get_frustum_vertices_camfr();

        var v0_camframe = vList[0].reshape(1,3);
        var v1_camframe = vList[1].reshape(1,3);
        var v2_camframe = vList[2].reshape(1,3);
        var v3_camframe = vList[3].reshape(1,3);
        var v4_camframe = vList[4].reshape(1,3);

        var v0_worldframe = cTw.transform_from(v0_camframe);
        var v1_worldframe = cTw.transform_from(v1_camframe);
        var v2_worldframe = cTw.transform_from(v2_camframe);
        var v3_worldframe = cTw.transform_from(v3_camframe);
        var v4_worldframe = cTw.transform_from(v4_camframe);

        return [v0_worldframe.tolist(), 
            v1_worldframe.tolist(), 
            v2_worldframe.tolist(), 
            v3_worldframe.tolist(), 
            v4_worldframe.tolist()];
    }

    calculate_scaled_ray_dirs(ray_dirs, frustum_ray_len) {
        /*Given frustum vertex ray directions and length, scale the rays to desired length.

        Args:
            ray_dirs: Array of shape (5,3) containing normalized vertex vectors.
            frustum_ray_len (float): Amount to scale vectors by.
        Returns:
            An array, shape (5,3), of the scaled frustum vertex vectors in camera coordinate frame.
        */

        // Center v0 at the origin.
        var v0_scaled = nj.array([0,0,0]);
        var vectors_list = [v0_scaled];

        // Scale v1...v4 by scalar: frustum_ray_len
        for (var row = 1; row < 5; row++) {
            const vector_scaled = nj.array([ray_dirs.get(row,0)*frustum_ray_len,
                                        ray_dirs.get(row,1)*frustum_ray_len,
                                        ray_dirs.get(row,2)*frustum_ray_len]);
            vectors_list.push(vector_scaled);
        }
        return vectors_list;
    }
}

module.exports = ViewFrustum;