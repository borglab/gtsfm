/* Unit Test for the ViewFrustum Class.

Author: Adi Singh
*/

// Third-Party Package Imports.
var nj = require('numjs');

// Local Imports.
var ViewFrustum = require('../Components/frustum_classes/view_frustum');

describe('ViewFrustum Class Unit Tests', () => {

    /* Creates sample frustum rays and checks if they are normalized properly. */
    it ('Test normalize_ray_dirs.', () => {
        const fx = 5;
        const img_w = 4;
        const img_h = 4;
        const ray_len = 0.5;
        const testFrustum = new ViewFrustum(fx, img_w, img_h, ray_len);
        
        var ray_dirs = nj.array([[0,0,2],
                                 [3,0,0],
                                 [0,4,0],
                                 [0,3,4],
                                 [4,4,2]]);
        const ray_dirs_normalized = testFrustum.normalize_ray_dirs(ray_dirs);

        const expected_ray_dirs = nj.array([[0,0,1],
                                           [1,0,0],
                                           [0,1,0],
                                           [0,0.6,0.8],
                                           [(2/3),(2/3),(1/3)]]);
        expect(expected_ray_dirs).toEqual(ray_dirs_normalized);
    });
    
    /* Creates a sample uv matrix and checks if the correct ray directions are outputted in normalized form. */
    it('Test compute_pixel_ray_directions_vectorized.', () => {
        const uv = nj.array([[0,0], [1,1], [2,2], [3,3], [4,4]]);
        const img_w = 4;
        const img_h = 4;
        const fx = 5;
        const ray_len = 0.5;

        const testFrustum = new ViewFrustum(fx, img_w, img_h, ray_len);

        const ray_dirs = testFrustum.compute_pixel_ray_directions_vectorized(uv, img_w, img_h);

        // Ray dirs not normalized.
        var expected_ray_dirs = nj.array([[-2,-2,5],
                                          [-1,-1,5],
                                          [0,0,5],
                                          [1,1,5],
                                          [2,2,5]]);
        
        // Normalizing ray dirs.
        for (var row = 0; row < expected_ray_dirs.shape[0]; row++) {
            var vector_mag = Math.sqrt(Math.pow(expected_ray_dirs.get(row,0),2) + 
                                        Math.pow(expected_ray_dirs.get(row,1),2) + 
                                        Math.pow(expected_ray_dirs.get(row,2),2));
            for (var col = 0; col < ray_dirs.shape[1]; col++) {
                expected_ray_dirs.set(row, col, (expected_ray_dirs.get(row,col)/vector_mag));
            }
        }
        expect(expected_ray_dirs).toEqual(ray_dirs);
    });

    /* Initializes a test ViewFrustum object and checks if the outputted vertices in the camera coordinate
       frame are correct.
     */
    it ('Test get_frustum_vertices_camfr.', () => {
        const img_w = 4;
        const img_h = 4;
        const fx = 5;
        const ray_len = 0.5;

        const testFrustum = new ViewFrustum(fx, img_w, img_h, ray_len);
        const vertices = testFrustum.get_frustum_vertices_camfr(); 
        
        const expected_vertices = [nj.array([0,0,0]),
                                   nj.array([(-1/Math.sqrt(33)), (-1/Math.sqrt(33)), (5/(2*Math.sqrt(33)))]),
                                   nj.array([(1/(2*Math.sqrt(30))), (-1/Math.sqrt(30)), (5/(2*Math.sqrt(30)))]),
                                   nj.array([(1/(2*Math.sqrt(27))), (1/(2*Math.sqrt(27))), (5/(2*Math.sqrt(27)))]),
                                   nj.array([(-1/Math.sqrt(30)), (1/(2*Math.sqrt(30))), (5/(2*Math.sqrt(30)))])];
        
        expect(expected_vertices).toEqual(vertices);
    });

    /* Creates some sample ray directions and checks if they are scaled properly according to the scalar: ray_len. */
    it ('Test calculate_scaled_ray_dirs.', () => {
        const img_w = 4;
        const img_h = 4;
        const fx = 5;
        const ray_len = 0.5;

        const testFrustum = new ViewFrustum(fx, img_w, img_h, ray_len);
        const ray_dirs = nj.array([[1,1,1],
                                  [2,2,2],
                                  [3,3,3],
                                  [4,4,4],
                                  [5,5,5]]);
        const scaled_ray_dirs = testFrustum.calculate_scaled_ray_dirs(ray_dirs, ray_len);
        const expected_scaled_ray_dirs = [nj.array([0,0,0]), 
                                          nj.array([1,1,1]),
                                          nj.array([1.5,1.5,1.5]),
                                          nj.array([2,2,2]),
                                          nj.array([2.5,2.5,2.5])];
        expect(expected_scaled_ray_dirs).toEqual(scaled_ray_dirs);
    })
})