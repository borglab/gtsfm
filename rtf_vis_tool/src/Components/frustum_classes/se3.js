/*Contains the Pose Matrix used to convert points from camera coordinate system to world coordinate system.

Author: Adi Singh
*/

// Third-Party Package Imports.
var nj = require('numjs');

class SE3 {
    // An SE3 class allows point cloud rotation and translation operations.

    constructor(rotation, translation) {
        /*Initialize an SE3 instance and pose matrix with its rotation and translation matrices.

        Args:
            rotation: Matrix of shape 3x3
            translation: Array of length 3
        */
        if (rotation.shape[0] !== 3 || rotation.shape[1] !== 3) throw new Error('Invalid Rotation Matrix');
        if (translation.shape[0] !== 3) throw new Error('Invalid Translation Matrix');
        this.rotation = rotation;
        this.translation = translation;
        this.transform_matrix = nj.identity(4);

        //set upper left 3x3 to rotation matrix
        for (var row = 0; row < 3; row++) {
            for (var col = 0; col < 3; col++) {
                this.transform_matrix.set(row, col, rotation.get(row,col));
            }

            //set last column to translation array
            this.transform_matrix.set(row, 3, this.translation.get(row));
        }
    }

    transform_from(point) {
        /*Apply the SE3 transformation to the point.

        Args:
            point: Array of shape (1,3) representing the point in the c frame.
        Returns:
            Array, shape (1,3), representing the point in the w frame.
        */
        var ones = nj.ones([1, 1]);
        var homogeneous_pt = nj.concatenate(point, ones);

        var point_world_coords = homogeneous_pt.dot(this.transform_matrix);

        point_world_coords = nj.array([[point_world_coords.get(0,0),
                            point_world_coords.get(0,1),
                            point_world_coords.get(0,2)]])
        return point_world_coords;
    }
}

module.exports = SE3;