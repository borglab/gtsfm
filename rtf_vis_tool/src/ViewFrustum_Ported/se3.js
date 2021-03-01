var nj = require('numjs');

module.exports = class SE3 {
    //An SE3 class allows point cloud rotation and translation operations.

    constructor(rotation, translation) {
        /*
            Initialize an SE3 instance with its rotation and translation matrices

            Args:
                rotation: Array of shape (3,3)
                translation: Array of shape (3,)
        */
        if (rotation.shape[0] !== 3 || rotation.shape[1] !== 3) throw 'Invalid Rotation Matrix';
        if (translation.shape[0] !== [3]) throw 'Invalid Translation Matrix';
        this.rotation = rotation;
        this.translation = translation;

        this.transform_matrix = nj.identity(4);
        this.transform_matrix.set(0,0,this.rotation.get(0,0));
        this.transform_matrix.set(0,1,this.rotation.get(0,1));
        this.transform_matrix.set(0,2,this.rotation.get(0,2));
        this.transform_matrix.set(1,0,this.rotation.get(1,0));
        this.transform_matrix.set(1,1,this.rotation.get(1,1));
        this.transform_matrix.set(1,2,this.rotation.get(1,2));
        this.transform_matrix.set(2,0,this.rotation.get(2,0));
        this.transform_matrix.set(2,1,this.rotation.get(2,1));
        this.transform_matrix.set(2,2,this.rotation.get(2,2));
        this.transform_matrix.set(0,3,this.translation.get(0));
        this.transform_matrix.set(1,3,this.translation.get(1));
        this.transform_matrix.set(2,3,this.translation.get(2));
    }

    transform_point_cloud(point_cloud) {
        /*
            Apply the SE3 transformation to the point cloud.

            Args:
                point_cloud: Array of shape (N,3)
            Returns:
                trasnformed_point_cloud: Array of shape (N,3)
        */
        var num_pts = point_cloud.shape[0];
        var ones = nj.ones([num_pts, 1]);
        var homogeneous_pts = nj.concatenate(point_cloud, ones);

        var transformed_point_cloud = homogeneous_pts.dot(this.transform_matrix.T);

        var sliced = nj.array([[transformed_point_cloud.get(0,0),
                            transformed_point_cloud.get(0,1),
                            transformed_point_cloud.get(0,2)]])
        return sliced;
    }
}