/* Unit Test for the SE3 Class.

Author: Adi Singh
*/

// Third-Party Package Imports.
var nj = require('numjs');

// Local Imports.
var SE3 = require('../Components/frustum_classes/se3');

describe('SE3 Class Unit Tests', () => {

    /* Creates a simple Pose matrix that flips x and z coordinates and translates by [2,3,4]. 
       Applies to the sample point [1,2,3] and verifies the SE3 transformation result is accurate.
    */
    it('Test transform_from.', () => {
        const rotation = nj.array([[0, 0, 1],
                                   [0, 1, 0],
                                   [1, 0, 0]]);
        const translation = nj.array([2,3,4]);
        const testSE3 = SE3(rotation, translation);
        const point = nj.array([1, 2, 3]);

        const computedPoint = testSE3.transform_from(point);
        const expectedPoint = nj.array([[5, 5, 5]]);

        expected(expectedPoint).toEqual(computedPoint);
    })
});

